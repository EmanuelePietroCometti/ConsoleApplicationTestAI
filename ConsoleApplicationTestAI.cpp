#include "Windows.h"
#include <timeapi.h>      // timeBeginPeriod / timeEndPeriod
#include "conio.h"
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <optional>
#include <utility>
#include <tchar.h>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <cwctype>
#include <climits>
#include <fmt/core.h>
#include <fmt/color.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

#pragma comment(lib, "winmm.lib")   // timeBeginPeriod / timeEndPeriod

// Default 1: da variabile globale non inizializzata valeva 0, quindi senza il
// 7° argomento 'c' configurava zero punti e 's' non avviava alcun thread,
// facendo sembrare il programma bloccato.
int num_control_points = 1; // numero di control points da simulare

enum class InferenceState : WORD {
	IDLE = 0,
	RESULT_READY = 1,
	PENDING = 2,
	ERROR_DETECTED = 4
};
enum class PointState : WORD {
	IDLE = 0,
	CONFIGURED = 1,
	UPDATE_PENDING = 2,
	QUIT = 3,
	ERROR_DETECTED = 4
};
enum class ListState : WORD {
	IDLE = 0,
	CONFIGURED = 1,
	UPDATE_PENDING = 2,
	QUIT = 3,
	ERROR_DETECTED = 4
};
enum class InferenceType : WORD {
	ANOMALY = 0,
	CLASSIFICATION = 1,
	OBJECT_DETECTION = 2
};

typedef struct resultInference {
	InferenceState state;
	DWORD sizeX;
	DWORD sizeY;
	TCHAR json[1024];
} resultInference, * PTresultInference;

typedef struct controlPoint {
	DWORD idPunto;
	DWORD sizeX;
	DWORD sizeY;
	DWORD bpp;
	TCHAR pathModello[512];
	TCHAR mutexName[128];
	TCHAR resultsEventName[128];
	TCHAR eventReadyName[128];
	PointState status;
	resultInference results;
	InferenceType inferenceType;
} controlPoint, * PTcontrolPoint;

typedef struct controlPointsList {
	DWORD numPunti;
	ListState state;
	TCHAR listMutexName[128];
	TCHAR listEventTriggerName[128];
	TCHAR listEventAckName[128];
	controlPoint points[1024];
} controlPointsList, * PTcontrolPointsList;

struct ScopedMutex {
	HANDLE hMutex;
	ScopedMutex(HANDLE mtx) : hMutex(mtx) {}
	~ScopedMutex() {
		if (hMutex != NULL) {
			ReleaseMutex(hMutex);
		}
	}
};

// ============================================================================
//  ASYNC LOGGER
//  Tutto l'output su console passa da un unico thread consumatore. Il percorso
//  caldo (thread frame-grabber) fa solo fmt::format + push in coda sotto un
//  mutex tenuto per pochi microsecondi: NESSUNA scrittura su console nel loop
//  temporizzato. Questo elimina:
//    - il blocco da console-lock globale di Windows dentro la sezione temporizzata
//    - il feedback loop "piu' drop -> piu' stampe -> piu' jitter -> piu' drop"
//  Un solo writer garantisce inoltre righe non interlacciate tra thread.
// ============================================================================
class AsyncLogger {
public:
	AsyncLogger() : running_(true) {
		worker_ = std::thread(&AsyncLogger::run, this);
	}
	~AsyncLogger() { stop(); }

	AsyncLogger(const AsyncLogger&) = delete;
	AsyncLogger& operator=(const AsyncLogger&) = delete;

	void log(std::string text, bool toStderr = false,
		std::optional<fmt::color> color = std::nullopt) {
			{
				std::lock_guard<std::mutex> lk(mtx_);
				queue_.push_back(Entry{ std::move(text), toStderr, color });
			}
			cv_.notify_one();
	}

	// Svuota la coda e ferma il worker. Idempotente: la seconda chiamata
	// (es. dal distruttore statico) esce subito.
	void stop() {
		if (!running_.exchange(false)) return;
		cv_.notify_one();
		if (worker_.joinable()) worker_.join();
	}

private:
	struct Entry {
		std::string text;
		bool toStderr;
		std::optional<fmt::color> color;
	};

	void run() {
		// Il logger non deve mai competere con il thread trigger per la CPU.
		SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
		std::unique_lock<std::mutex> lk(mtx_);
		for (;;) {
			cv_.wait(lk, [&] { return !queue_.empty() || !running_.load(); });
			while (!queue_.empty()) {
				Entry e = std::move(queue_.front());
				queue_.pop_front();
				lk.unlock();
				FILE* out = e.toStderr ? stderr : stdout;
				if (e.color) fmt::print(out, fg(*e.color), "{}", e.text);
				else         fmt::print(out, "{}", e.text);
				lk.lock();
			}
			if (!running_.load()) break;   // coda vuota e in fase di stop
		}
	}

	std::mutex mtx_;
	std::condition_variable cv_;
	std::deque<Entry> queue_;
	std::thread worker_;
	std::atomic<bool> running_;
};

// Meyers singleton: costruito alla prima chiamata (dentro wmain), distrutto
// all'uscita del processo. Evita problemi di ordine di init statico.
static AsyncLogger& logger() {
	static AsyncLogger inst;
	return inst;
}

// fmt emette i colori come sequenze ANSI (\x1b[38;2;...m): il conhost classico
// non le interpreta finche' non si attiva ENABLE_VIRTUAL_TERMINAL_PROCESSING,
// e le stampa letteralmente come "<-[38;2;...m". Se l'attivazione fallisce
// (console troppo vecchia) i colori vengono semplicemente disabilitati.
static bool g_ansiColorSupported = false;

static void EnableVTProcessing() {
	g_ansiColorSupported = true;
	for (DWORD stdHandle : { STD_OUTPUT_HANDLE, STD_ERROR_HANDLE }) {
		HANDLE h = GetStdHandle(stdHandle);
		DWORD mode = 0;
		if (h == NULL || h == INVALID_HANDLE_VALUE || !GetConsoleMode(h, &mode)) {
			continue;   // stream redirezionato su file/pipe: non e' una console
		}
		if (!SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING)) {
			g_ansiColorSupported = false;
		}
	}
}

// Helper: formattano il messaggio e lo accodano. Sostituiscono fmt::print.
template <typename... Args>
static void LOG(fmt::format_string<Args...> f, Args&&... a) {
	logger().log(fmt::format(f, std::forward<Args>(a)...), /*stderr*/ false);
}
template <typename... Args>
static void LOGE(fmt::format_string<Args...> f, Args&&... a) {
	logger().log(fmt::format(f, std::forward<Args>(a)...), /*stderr*/ true);
}
template <typename... Args>
static void LOGC(fmt::color c, bool toStderr, fmt::format_string<Args...> f, Args&&... a) {
	logger().log(fmt::format(f, std::forward<Args>(a)...), toStderr,
		g_ansiColorSupported ? std::optional<fmt::color>(c) : std::nullopt);
}

// Runtime configuration (overridable via command line, see wmain)
std::filesystem::path g_inputFolder = L"D:\\emanuele\\Code\\ConsoleApplicationTestAI\\dataset";
std::wstring g_modelPath = L"D:\\emanuele\\Code\\ConsoleApplicationTestAI\\model\\yolo_pure.onnx";
InferenceType g_inferenceType = InferenceType::CLASSIFICATION;

HANDLE hcpListMMF = NULL;
PTcontrolPointsList cpListMMF = NULL;

std::vector<std::thread> threads;
std::vector<HANDLE> hControlPointMutex;
std::vector<HANDLE> hControlPointEvent;
std::vector<HANDLE> hControlPointResults;

HANDLE hListMutex = NULL;
HANDLE hListEventTrigger = NULL;
HANDLE hListEventAck = NULL;
int frameRate_milliseconds = 50;   // intervallo del frame grabber simulato
int g_triggerCore = -1;   // logical core for the trigger thread (-1 = no pinning)

// Statistiche per control point. Il tempo di attesa dell'ack (trigger ->
// RESULT_READY) e' misurato solo sui risultati validi. Scritte dal thread del
// control point prima di terminare, lette dal main dopo la join.
struct WaitStats {
	unsigned long framesSent = 0;   // frame inviati (trigger emessi)
	unsigned long acksReceived = 0; // risultati validi ricevuti (attese misurate)
	unsigned long frameDrops = 0;   // tick con inferenza ancora PENDING
	unsigned long missedTicks = 0;  // risvegli del simulatore in ritardo >= 1 intervallo
	long long totalWaitUs = 0;
	long long minWaitUs = 0;
	long long maxWaitUs = 0;
};
std::vector<WaitStats> g_stats;

// Convert wide strings coming from the IPC structs / command line so all
// logging can go through narrow fmt::print (avoids mixing stream orientations)
static std::string ToNarrow(const std::wstring& wide) {
	return std::filesystem::path(wide).string();
}

void controlPointThreadFunc(int i)
{
	// Il processo ONNX gira in REALTIME con thread TIME_CRITICAL in spin sui
	// core del suo slice: il produttore deve avere priorita' massima nella
	// propria classe e, se richiesto, essere pinnato su un core FUORI dallo
	// slice ONNX, altrimenti viene starvato e i risvegli in ritardo si
	// trasformano in falsi frame drop.
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
	if (g_triggerCore >= 0) {
		if (SetThreadAffinityMask(GetCurrentThread(), 1ULL << g_triggerCore) == 0) {
			LOGE("### WARNING: SetThreadAffinityMask({}) failed ({})\n", g_triggerCore, GetLastError());
		}
		else {
			LOG("Trigger thread {} pinned to logical core {}\n", i, g_triggerCore);
		}
	}

	// Map INPUT IMAGE (Sender -> AI)
	int nPayload = cpListMMF->points[i].sizeX * cpListMMF->points[i].sizeY * cpListMMF->points[i].bpp / 8;
	std::wstring pName = L"MMF_" + std::to_wstring(cpListMMF->points[i].idPunto) + L"_IMAGE";
	HANDLE hMMFImage = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, nPayload, pName.c_str());
	void* pImage = NULL;
	if (hMMFImage != NULL) {
		pImage = MapViewOfFile(hMMFImage, FILE_MAP_WRITE, 0, 0, nPayload);
	}

	// Map OUTPUT RESIMAGE (AI -> Sender)
	// Must match the size the worker allocates (sizeX * sizeY * bpp/8):
	// the heatmap is written back as RGB, not grayscale
	int nResPayload = cpListMMF->points[i].sizeX * cpListMMF->points[i].sizeY * cpListMMF->points[i].bpp / 8;
	std::wstring resName = L"MMF_" + std::to_wstring(cpListMMF->points[i].idPunto) + L"_RESIMAGE";
	HANDLE hMMFResImage = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, nResPayload, resName.c_str());
	void* pResImage = NULL;
	if (hMMFResImage != NULL) {
		pResImage = MapViewOfFile(hMMFResImage, FILE_MAP_READ, 0, 0, nResPayload);
	}

	// PRELOAD: decodifica + resize di TUTTE le immagini una volta sola, PRIMA
	// del loop temporizzato. La versione precedente collezionava solo i path e
	// faceva imread+resize a ogni ciclo: quando il decode sforava il tempo
	// morto tra due tick, il thread arrivava in ritardo e produceva raffiche
	// di falsi drop. Cosi' nel loop resta solo un memcpy (~0.1 ms a frame).
	std::vector<cv::Mat> frames;
	std::vector<std::string> frameNames;

	try {
		for (const auto& entry : std::filesystem::recursive_directory_iterator(g_inputFolder)) {
			if (entry.is_regular_file()) {
				std::string ext = entry.path().extension().string();
				std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
				if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
					std::string filename = entry.path().filename().string();
					cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
					if (img.empty()) {
						LOGE("### ERROR: Failed to decode {}\n", filename);
						continue;
					}
					// Force resize to match the Shared Memory allocated layout
					cv::Mat resized;
					cv::resize(img, resized, cv::Size(cpListMMF->points[i].sizeX, cpListMMF->points[i].sizeY));
					if (!resized.isContinuous()) {
						resized = resized.clone();
					}
					frames.push_back(std::move(resized));
					frameNames.push_back(std::move(filename));
				}
			}
		}
	}
	catch (const std::exception& e) {
		LOGE("### ERROR reading directory: {}\n", e.what());
	}

	LOG("Thread {} running (Simulating Frame Grabber @ {} ms). Preloaded {} frames ({} MB in RAM).\n",
		i, frameRate_milliseconds, frames.size(), frames.size() * static_cast<size_t>(nPayload) / (1024 * 1024));

	int imageIndex = 0;
	std::string lastSentFilename = "NONE";

	WaitStats stats;
	stats.minWaitUs = LLONG_MAX;

	// Copies the current preloaded frame into shared memory and triggers the
	// AI engine. Must be called with hControlPointMutex[i] held.
	auto sendStagedFrame = [&]() {
		if (pImage != NULL) {
			if (!frames.empty()) {
				memcpy(pImage, frames[imageIndex].data, nPayload);
				lastSentFilename = frameNames[imageIndex];
				imageIndex = (imageIndex + 1) % static_cast<int>(frames.size());
			}
			else {
				// Fallback mechanism if the folder is empty or invalid
				uint8_t* pPixels = static_cast<uint8_t*>(pImage);
				for (int p = 0; p < nPayload; p++) {
					pPixels[p] = static_cast<uint8_t>(p % 2);
				}
			}
		}

		// TRIGGER AI ENGINE FOR THE NEW FRAME
		cpListMMF->points[i].results.state = InferenceState::PENDING;
		stats.framesSent++;
		ResetEvent(hControlPointResults[i]);
		SetEvent(hControlPointEvent[i]);
		};

	// Parsing e stampa del risultato del frame: chiamato FUORI dal lock, su
	// copie locali (la memoria condivisa puo' essere riscritta dal worker
	// subito dopo il release). Aggiorna anche le statistiche di attesa: il
	// timer viene conteggiato SOLO su RESULT_READY.
	auto reportResult = [&](InferenceState prevState, InferenceType prevType,
		const wchar_t* jsonCopy, const std::string& ackedFilename, long long waitUs) {

		if (prevState == InferenceState::RESULT_READY) {

			stats.acksReceived++;
			stats.totalWaitUs += waitUs;
			if (waitUs < stats.minWaitUs) stats.minWaitUs = waitUs;
			if (waitUs > stats.maxWaitUs) stats.maxWaitUs = waitUs;

			// The JSON payload is plain ASCII: convert once and parse narrow
			const wchar_t* p = jsonCopy;
			std::string jsonStr;
			jsonStr.reserve(wcsnlen(p, 1024));
			for (size_t k = 0; k < 1024 && p[k]; ++k)
				jsonStr.push_back(static_cast<char>(p[k]));

			if (jsonStr.empty()) {
				LOGE("CP {} | FILE: {} | RESULT_READY but empty JSON payload\n", i, ackedFilename);
				return;
			}

			// One try around ALL access to j: parse_error, type_error and out_of_range
			// all derive from json::exception. Without this, a type_error would escape
			// the thread and terminate the process.
			try {
				const json j = json::parse(jsonStr);

				if (!j.is_object()) {
					LOGE("CP {} | FILE: {} | JSON is not an object: {}\n", i, ackedFilename, jsonStr);
					return;
				}

				if (prevType == InferenceType::ANOMALY) {
					const float anomalyScore = j.value("anomaly_score", -1.0f);
					const std::string status = j.value("status", std::string("UNKNOWN"));

					LOG(">> CP {} | FILE: {} | ANOMALY_SCORE: {:.6f} | STATUS: {} | ACK WAIT: {:.3f} ms\n",
						i, ackedFilename, anomalyScore, status, waitUs / 1000.0);
				}
				else if (prevType == InferenceType::CLASSIFICATION) {
					const float confidence = j.value("confidence", -1.0f);

					// class_id is a label string ("1_BadPins"); accept a numeric value
					// too, in case the producer ever switches to a plain index.
					std::string class_id = "UNKNOWN";
					if (const auto it = j.find("class_id"); it != j.end()) {
						class_id = it->is_string() ? it->get<std::string>() : it->dump();
					}

					LOG(">> CP {} | FILE: {} | CLASS_ID: {} | CONFIDENCE: {:.6f} | ACK WAIT: {:.3f} ms\n",
						i, ackedFilename, class_id, confidence, waitUs / 1000.0);
				}
				else {
					// OBJECT_DETECTION or unhandled type: raw dump is better than nothing
					LOG(">> CP {} | FILE: {} | RAW: {} | ACK WAIT: {:.3f} ms\n",
						i, ackedFilename, jsonStr, waitUs / 1000.0);
				}
			}
			catch (const json::exception& e) {
				LOGE("CP {} | FILE: {} | JSON error ({}): {} | payload: {}\n",
					i, ackedFilename, e.id, e.what(), jsonStr);
			}
		}
	};

	// FRAME GRABBER SIMULATO: trigger a intervallo fisso (default 50 ms), come
	// una telecamera reale. Tra un tick e l'altro il thread resta in ascolto
	// dell'ack: se l'inferenza finisce prima, scarica subito la MMF e riporta
	// lo stato a IDLE, poi attende lo scadere dell'intervallo. Al tick pusha
	// solo se lo stato e' IDLE o RESULT_READY; se e' ancora PENDING la
	// telecamera non puo' consegnare il frame: frame drop.
	const auto hardwareTriggerInterval = std::chrono::milliseconds(frameRate_milliseconds);
	constexpr auto kSpinWindow = std::chrono::milliseconds(2);

	std::chrono::steady_clock::time_point triggerTime;
	bool keepRunning = true;

	// Primo frame: fissa anche l'origine della griglia dei tick.
	{
		DWORD dwWaitResult = WaitForSingleObject(hControlPointMutex[i], INFINITE);
		if (dwWaitResult == WAIT_OBJECT_0 || dwWaitResult == WAIT_ABANDONED) {
			if (cpListMMF->points[i].status == PointState::QUIT) {
				keepRunning = false;
			}
			else {
				sendStagedFrame();
				triggerTime = std::chrono::steady_clock::now();
			}
			ReleaseMutex(hControlPointMutex[i]);
		}
	}
	auto next_trigger_time = std::chrono::steady_clock::now() + hardwareTriggerInterval;

	while (keepRunning)
	{
		// FASE 1: ascolto dell'ack fino a ridosso del prossimo tick
		// WaitForSingleObject con timeout fa sia da sleep sia da ascolto: se
		// l'inferenza finisce prima del tick, il risultato viene scaricato
		// subito e lo slot torna IDLE, pronto per il push al tick.
		for (;;) {
			const auto now = std::chrono::steady_clock::now();
			if (now >= next_trigger_time - kSpinWindow) break;
			const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
				next_trigger_time - kSpinWindow - now);
			const DWORD timeoutMs = static_cast<DWORD>(std::max<long long>(1, remaining.count()));

			if (WaitForSingleObject(hControlPointResults[i], timeoutMs) != WAIT_OBJECT_0) {
				continue;   // timeout: ricontrolla quanto manca al tick
			}
			const auto ackTime = std::chrono::steady_clock::now();

			InferenceState prevState = InferenceState::IDLE;
			InferenceType  prevType = g_inferenceType;
			wchar_t        jsonCopy[1024] = { 0 };
			long long      waitUs = 0;
			const std::string ackedFilename = lastSentFilename;

			DWORD dwWaitResult = WaitForSingleObject(hControlPointMutex[i], INFINITE);
			if (dwWaitResult == WAIT_OBJECT_0 || dwWaitResult == WAIT_ABANDONED) {
				if (cpListMMF->points[i].status == PointState::QUIT) {
					keepRunning = false;
				}
				else if (cpListMMF->points[i].results.state == InferenceState::RESULT_READY) {
					prevState = InferenceState::RESULT_READY;
					prevType = cpListMMF->points[i].inferenceType;
					memcpy(jsonCopy, cpListMMF->points[i].results.json, sizeof(jsonCopy));
					// Timer fermato SOLO su risultato valido.
					waitUs = std::chrono::duration_cast<std::chrono::microseconds>(
						ackTime - triggerTime).count();
					// MMF scaricata: lo slot torna libero per il prossimo push.
					cpListMMF->points[i].results.state = InferenceState::IDLE;
				}
				ReleaseMutex(hControlPointMutex[i]);
			}
			if (!keepRunning) break;

			reportResult(prevState, prevType, jsonCopy, ackedFilename, waitUs);
		}
		if (!keepRunning) break;

		// Spin finale: precisione al microsecondo sull'istante del tick.
		while (std::chrono::steady_clock::now() < next_trigger_time) {
			YieldProcessor();
		}

		const auto now = std::chrono::steady_clock::now();
		next_trigger_time += hardwareTriggerInterval;
		// Tick persi per risveglio in ritardo >= 1 intervallo intero: saltati,
		// non recuperati in raffica (una camera reale non emette mai due
		// trigger back-to-back). Sono ritardi del simulatore, non dell'AI.
		while (next_trigger_time <= now) {
			next_trigger_time += hardwareTriggerInterval;
			stats.missedTicks++;
		}

		// FASE 2: tick del frame grabber, tentativo di push
		bool frameDropped = false;
		InferenceState prevState = InferenceState::IDLE;
		InferenceType  prevType = g_inferenceType;
		wchar_t        jsonCopy[1024] = { 0 };
		long long      waitUs = 0;
		const std::string ackedFilename = lastSentFilename;

		DWORD dwWaitResult = WaitForSingleObject(hControlPointMutex[i], INFINITE);
		if (dwWaitResult == WAIT_OBJECT_0 || dwWaitResult == WAIT_ABANDONED)
		{
			if (cpListMMF->points[i].status == PointState::QUIT) {
				keepRunning = false;
			}
			else if (cpListMMF->points[i].results.state == InferenceState::PENDING) {
				// L'inferenza del frame precedente non ha ancora finito: la
				// telecamera non puo' consegnare il frame, drop.
				stats.frameDrops++;
				frameDropped = true;
			}
			else
			{
				// Stato IDLE (risultato gia' scaricato in fase 1) oppure
				// RESULT_READY (arrivato a ridosso del tick, non ancora
				// scaricato): in quel caso lo si copia PRIMA del push.
				prevState = cpListMMF->points[i].results.state;
				prevType = cpListMMF->points[i].inferenceType;
				if (prevState == InferenceState::RESULT_READY) {
					memcpy(jsonCopy, cpListMMF->points[i].results.json, sizeof(jsonCopy));
					waitUs = std::chrono::duration_cast<std::chrono::microseconds>(
						now - triggerTime).count();
				}

				sendStagedFrame();
				triggerTime = std::chrono::steady_clock::now();
			}
			ReleaseMutex(hControlPointMutex[i]);
		}

		if (frameDropped) {
			LOGC(fmt::color::yellow, /*stderr*/ true,
				"### WARNING: FRAME DROP DETECTED ON CP {}! AI still PENDING at trigger tick (drops so far: {}).\n",
				i, stats.frameDrops);
		}
		if (!keepRunning) break;

		reportResult(prevState, prevType, jsonCopy, ackedFilename, waitUs);
	}

	if (pImage) UnmapViewOfFile(pImage);
	if (hMMFImage) CloseHandle(hMMFImage);
	if (pResImage) UnmapViewOfFile(pResImage);
	if (hMMFResImage) CloseHandle(hMMFResImage);

	if (stats.acksReceived == 0) stats.minWaitUs = 0;
	g_stats.emplace_back(stats);   // la join in wmain sincronizza la lettura

	LOG("Thread {} stopped! (frames sent: {}, results: {}, frame drops: {}, missed ticks: {})\n",
		i, stats.framesSent, stats.acksReceived, stats.frameDrops, stats.missedTicks);
}

void Quit() {
	DWORD mutexWait = WaitForSingleObject(hListMutex, INFINITE);
	switch (mutexWait) {
		// WAIT_ABANDONED grants ownership too and must release the mutex like
		// WAIT_OBJECT_0, otherwise it leaks (see the same handling in Configure)
	case WAIT_ABANDONED:
	case WAIT_OBJECT_0:
	{
		__try {
			cpListMMF->state = ListState::QUIT;
			for (int i = 0; i < num_control_points; i++) {
				DWORD innerMutexWait = WaitForSingleObject(hControlPointMutex[i], INFINITE);
				if (innerMutexWait == WAIT_OBJECT_0 || innerMutexWait == WAIT_ABANDONED) {
					cpListMMF->points[i].status = PointState::QUIT;
				}
				ReleaseMutex(hControlPointMutex[i]);

				// Send a message to the control point thread to notify the status update 
				SetEvent(hControlPointEvent[i]);
				SetEvent(hControlPointResults[i]);
			}
		}
		__finally {
			ReleaseMutex(hListMutex);
		}

		SetEvent(hListEventTrigger);

		DWORD ackWait = WaitForSingleObject(hListEventAck, 3000);
		if (ackWait == WAIT_OBJECT_0) {
			LOG("ONNX manager shutdown acknowledged.\n");
		}
		else {
			LOG("### ONNX manager shutdown timeout!\n");
		}
		break;
	}
	case WAIT_TIMEOUT:
	{
		LOG("Main thread waiting for the global mutex!\n");
		break;
	}
	}
}

void Start()
{
	static bool isStarted = false;
	if (isStarted) {
		LOG("### WARNING: Simulation already running!\n");
		return;
	}
	isStarted = true;
	for (int i = 0; i < num_control_points; i++)
	{
		threads.emplace_back(std::thread(controlPointThreadFunc, i));
		std::this_thread::sleep_for(std::chrono::milliseconds(30));
	}
}

void createMutex(HANDLE& hMutex, int i) {
	if (hMutex != NULL && hMutex != INVALID_HANDLE_VALUE) {
		CloseHandle(hMutex);
		hMutex = NULL;
	}

	TCHAR name[128];
	swprintf_s(name, 128, TEXT("CP_%lu_MUTEX"), cpListMMF->points[i].idPunto);

	hMutex = CreateMutex(NULL, FALSE, name);

	if (hMutex == NULL) {
		LOGE("CreateMutex error: {}\n", GetLastError());
	}
	else {
		LOG("CreateMutex {}: {}\n",
			(GetLastError() == ERROR_ALREADY_EXISTS) ? "opened existing" : "created new", ToNarrow(name));
	}
}

void createEvent(HANDLE& hEvent, int i, const std::wstring& suffix) {
	if (hEvent != NULL && hEvent != INVALID_HANDLE_VALUE) {
		CloseHandle(hEvent);
		hEvent = NULL;
	}

	TCHAR name[128];
	if (suffix == L"EVENTREADY") {
		swprintf_s(name, 128, TEXT("CP_%lu_EVENTREADY"), cpListMMF->points[i].idPunto);
	}
	else {
		swprintf_s(name, 128, TEXT("CP_%lu_EVENTRESULTS"), cpListMMF->points[i].idPunto);
	}

	hEvent = CreateEvent(NULL, FALSE, FALSE, name);

	if (hEvent == NULL) {
		LOGE("CreateEvent error: {}\n", GetLastError());
	}
	else {
		LOG("CreateEvent {}: {}\n",
			(GetLastError() == ERROR_ALREADY_EXISTS) ? "opened existing" : "created new", ToNarrow(name));
	}
}

bool Configure()
{
	bool isConfigured = false;
	DWORD globalWaitResult = WaitForSingleObject(hListMutex, INFINITE);

	switch (globalWaitResult)
	{
		// WAIT_ABANDONED still grants ownership (previous owner crashed while holding
		// the mutex). It must be handled like WAIT_OBJECT_0 and released normally,
		// otherwise the mutex leaks and both processes end up deadlocked.
	case WAIT_ABANDONED:
	case WAIT_OBJECT_0:
	{
		cpListMMF->state = ListState::UPDATE_PENDING;

		ZeroMemory(cpListMMF->points, sizeof(controlPoint) * 1024);
		cpListMMF->numPunti = num_control_points;
		DWORD nRand = rand();

		for (int i = 0; i < num_control_points; i++)
		{
			cpListMMF->points[i].idPunto = nRand + i;
			cpListMMF->points[i].sizeX = 512;
			cpListMMF->points[i].sizeY = 512;
			cpListMMF->points[i].bpp = 24;

			wcscpy_s(cpListMMF->points[i].pathModello, 512, g_modelPath.c_str());

			swprintf_s(cpListMMF->points[i].mutexName, 128, TEXT("CP_%lu_MUTEX"), cpListMMF->points[i].idPunto);
			swprintf_s(cpListMMF->points[i].eventReadyName, 128, TEXT("CP_%lu_EVENTREADY"), cpListMMF->points[i].idPunto);
			swprintf_s(cpListMMF->points[i].resultsEventName, 128, TEXT("CP_%lu_EVENTRESULTS"), cpListMMF->points[i].idPunto);

			createMutex(hControlPointMutex[i], i);
			createEvent(hControlPointEvent[i], i, L"EVENTREADY");
			createEvent(hControlPointResults[i], i, L"EVENTRESULTS");

			cpListMMF->points[i].inferenceType = g_inferenceType;
			cpListMMF->points[i].status = PointState::UPDATE_PENDING;
		}

		LOG("Release the global mutex\n");
		if (!ReleaseMutex(hListMutex))
		{
			LOGE("### Release global mutex error\n");
		}

		// Notify the AI to start configuration (and TensorRT warmup)
		SetEvent(hListEventTrigger);

		LOG(">> Waiting for AI engine configuration...\n");
		LOG(">> [NOTE: First-time TensorRT optimization takes time. Press 'q' to abort]\n");

		bool ackReceived = false;
		bool abortRequested = false;

		// Continuous loop that keeps the UI unlocked
		while (!ackReceived && !abortRequested) {

			// Microscopic timeout to avoid CPU saturation
			DWORD ackWait = WaitForSingleObject(hListEventAck, 100);

			if (ackWait == WAIT_OBJECT_0) {
				// The AI has completed TensorRT warmup
				ackReceived = true;
			}
			else if (ackWait == WAIT_TIMEOUT) {
				// Constantly intercept keyboard input
				if (_kbhit()) {
					char c = _getch();
					if (c == 'q') {
						abortRequested = true;
					}
				}
			}
		}

		// If 'q' was pressed, interrupt cleanly and immediately
		if (abortRequested) {
			LOG("### WARNING: Configuration interrupted by user. Initiating shutdown...\n");
			Quit();
			return false;
		}

		// Update the global state
		WaitForSingleObject(hListMutex, INFINITE);
		if (cpListMMF->state == ListState::CONFIGURED) {
			LOG("Configuration control points completed!\n");
			isConfigured = true;
		}
		else {
			LOGE("### ERROR: control points configuration failed.\n");
		}
		ReleaseMutex(hListMutex);

		break;
	}
	case WAIT_TIMEOUT:
	{
		LOG("Main thread waiting for the global mutex!\n");
		break;
	}
	}
	return isConfigured;
}

// Called with hListMutex already owned; releases it in every case.
// Kept as a separate function because SEH (__try/__finally) cannot coexist
// with C++ objects requiring unwinding in the same function (C2712)
bool isQuitRequested()
{
	bool quit = false;
	__try {
		if (cpListMMF->state == ListState::QUIT) {
			quit = true;
		}
	}
	__finally {
		ReleaseMutex(hListMutex);
	}
	return quit;
}

// QuickEdit (conhost): un click nella finestra entra in modalita' selezione e
// congela le scritture su console; i tasti vanno alla selezione, non a _getch.
// Il processo sembra "bloccato" finche' non si preme ESC/Invio. Disattivarlo
// rende il freeze impossibile. ENABLE_EXTENDED_FLAGS e' obbligatorio, senza
// di esso la modifica a QUICK_EDIT viene ignorata.
static void DisableQuickEditMode() {
	HANDLE hIn = GetStdHandle(STD_INPUT_HANDLE);
	if (hIn == NULL || hIn == INVALID_HANDLE_VALUE) return;
	DWORD mode = 0;
	if (!GetConsoleMode(hIn, &mode)) return;   // stdin non e' una console
	mode &= ~ENABLE_QUICK_EDIT_MODE;
	mode |= ENABLE_EXTENDED_FLAGS;
	SetConsoleMode(hIn, mode);
}

// Parse the analysis type argument: accepts the enum name (case-insensitive) or its numeric value
bool parseInferenceType(std::wstring value, InferenceType& outType)
{
	std::transform(value.begin(), value.end(), value.begin(), ::towlower);

	if (value == L"anomaly" || value == L"0") {
		outType = InferenceType::ANOMALY;
	}
	else if (value == L"classification" || value == L"1") {
		outType = InferenceType::CLASSIFICATION;
	}
	else if (value == L"object_detection" || value == L"objectdetection" || value == L"2") {
		outType = InferenceType::OBJECT_DETECTION;
	}
	else {
		return false;
	}
	return true;
}

int wmain(int argc, wchar_t* argv[])
{
	// Va fatto PRIMA di qualunque stampa: elimina i freeze da click accidentale
	// e attiva l'interpretazione delle sequenze ANSI dei colori.
	DisableQuickEditMode();
	EnableVTProcessing();

	// Alza la risoluzione del timer di sistema a 1 ms per TUTTA la durata del
	// processo: migliora la granularita' sia degli sleep sia dei timeout di
	// WaitForSingleObject usati per attendere il tick del frame grabber.
	timeBeginPeriod(1);

	// Il processo ONNX gira in REALTIME_PRIORITY_CLASS con thread in spin:
	// senza alzare anche la nostra classe, il thread trigger viene starvato e
	// ogni risveglio in ritardo diventa un falso frame drop. HIGH (non
	// REALTIME: non serve battere l'AI, basta battere i processi normali,
	// e due processi REALTIME in competizione peggiorano entrambi).
	if (!SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS)) {
		LOGE("### WARNING: SetPriorityClass(HIGH) failed ({})\n", GetLastError());
	}

	LOG("DELTA VISIONE - Test AI via MMF + ONNX!\n");

	// Optional positional arguments; defaults defined at the top of the file
	// Usage: ConsoleApplicationTestAI.exe [inputFolder] [modelPath] [anomaly|classification|object_detection] [frameRateMs] [triggerCore]
	// triggerCore: logical core (0-based) where the trigger thread is pinned.
	// Pick a core OUTSIDE the ONNX slice (see the ONNX startup log); -1 = no pin.
	if (argc > 1) {
		g_inputFolder = argv[1];
	}
	if (argc > 2) {
		g_modelPath = argv[2];
		if (g_modelPath.length() >= 512) {
			LOGE("### ERROR: model path exceeds 511 characters.\n");
			logger().stop();
			timeEndPeriod(1);
			return 1;
		}
	}
	if (argc > 3) {
		if (!parseInferenceType(argv[3], g_inferenceType)) {
			LOGE("### ERROR: unknown analysis type '{}'. Valid values: anomaly (0), classification (1), object_detection (2).\n",
				ToNarrow(argv[3]));
			logger().stop();
			timeEndPeriod(1);
			return 1;
		}
	}

	if (argc > 4) {
		try {
			frameRate_milliseconds = std::stoi(argv[4]);
			if (frameRate_milliseconds <= 0) {
				LOGE("### ERROR: frame rate must be a positive number of milliseconds.\n");
				logger().stop();
				timeEndPeriod(1);
				return 1;
			}
		}
		catch (const std::exception&) {
			LOGE("### ERROR: invalid frame rate value: '{}'. Must be an integer.\n", ToNarrow(argv[4]));
			logger().stop();
			timeEndPeriod(1);
			return 1;
		}
	}

	if (argc > 5) {
		try {
			g_triggerCore = std::stoi(argv[5]);
			if (g_triggerCore > 63) {
				LOGE("### ERROR: trigger core must be < 64.\n");
				logger().stop();
				timeEndPeriod(1);
				return 1;
			}
		}
		catch (const std::exception&) {
			LOGE("### ERROR: invalid trigger core value: '{}'. Must be an integer (-1 = no pin).\n", ToNarrow(argv[5]));
			logger().stop();
			timeEndPeriod(1);
			return 1;
		}
	}

	if (argc > 6) {
		try {
			num_control_points = std::stoi(argv[6]);
			if (num_control_points <= 0) {
				LOGE("### ERROR: number of control points must be a positive number of milliseconds.\n");
				logger().stop();
				timeEndPeriod(1);
				return 1;
			}
		}
		catch (const std::exception&) {
			LOGE("### ERROR: invalid number of control points value: '{}'. Must be an integer.\n", ToNarrow(argv[6]));
			logger().stop();
			timeEndPeriod(1);
			return 1;
		}
	}



	if (!std::filesystem::is_directory(g_inputFolder)) {
		LOGE("### WARNING: input folder does not exist or is not a directory: {}\n", g_inputFolder.string());
	}
	if (!std::filesystem::is_regular_file(g_modelPath)) {
		LOGE("### WARNING: model file not found: {}\n", ToNarrow(g_modelPath));
	}

	LOG(">> Input folder : {}\n"
		">> Model path   : {}\n"
		">> Analysis type: {}\n"
		">> Trigger rate : {} ms\n"
		">> Trigger core : {}\n"
		">> Control pts  : {}\n\n",
		g_inputFolder.string(),
		ToNarrow(g_modelPath),
		g_inferenceType == InferenceType::ANOMALY ? "ANOMALY" :
		g_inferenceType == InferenceType::CLASSIFICATION ? "CLASSIFICATION" : "OBJECT_DETECTION",
		frameRate_milliseconds,
		g_triggerCore >= 0 ? std::to_string(g_triggerCore) : "not pinned",
		num_control_points);

	srand(1792);

	hListMutex = CreateMutex(NULL, FALSE, TEXT("LISTMUTEX"));
	hListEventTrigger = CreateEvent(NULL, FALSE, FALSE, TEXT("LISTEVENTTRIGGER"));
	hListEventAck = CreateEvent(NULL, FALSE, FALSE, TEXT("LISTEVENTACK"));


	hControlPointMutex.assign(num_control_points, 0);
	hControlPointEvent.assign(num_control_points, 0);
	hControlPointResults.assign(num_control_points, 0);

	hcpListMMF = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, sizeof(controlPointsList), L"CONTROLPOINTLIST");
	DWORD dwErr = GetLastError();
	if (dwErr == ERROR_SUCCESS || dwErr == ERROR_ALREADY_EXISTS)
		cpListMMF = (PTcontrolPointsList)MapViewOfFile(hcpListMMF, FILE_MAP_WRITE, 0, 0, sizeof(controlPointsList));
	if (cpListMMF != NULL)
	{
		cpListMMF->state = ListState::IDLE;
		wcscpy_s(cpListMMF->listMutexName, 128, TEXT("LISTMUTEX"));
		wcscpy_s(cpListMMF->listEventTriggerName, 128, TEXT("LISTEVENTTRIGGER"));
		wcscpy_s(cpListMMF->listEventAckName, 128, TEXT("LISTEVENTACK"));
	}
	else {
		LOGE("Could not map view of file ({}).\n", GetLastError());
		CloseHandle(hcpListMMF);
		logger().stop();
		timeEndPeriod(1);
		return 1;
	}

	LOG("Press 'q' to quit, press 'c' to configure, press 's' to start sim\n");

	bool isRunning = true;
	bool isConfigured = false;
	bool errorDetected = false;
	while (isRunning)
	{
		if (_kbhit())
		{
			char c = _getch();

			switch (c)
			{
			case 'c':
			{
				if (!isConfigured) {
					isConfigured = Configure();
				}
				if (!isConfigured) {
					LOGE("### ERROR: Configuration failed...starting shutdown!\n");
					errorDetected = true;
					Quit();
				}
				break;
			}
			case 's':
			{
				if (isConfigured) {
					Start();
				}
				else {
					LOGE("### ERROR: Configure before starting!\n");
				}
				break;
			}

			case 'q':
				LOG("Initiating shutdown sequence...\n");
				Quit();
				break;

			default:
				// Ignore any other key presses
				break;
			}
		}
		DWORD resWait = WaitForSingleObject(hListMutex, 100);
		switch (resWait) {
			// Su WAIT_ABANDONED il thread possiede comunque il mutex: va gestito
			// come WAIT_OBJECT_0 (isQuitRequested lo rilascia), altrimenti il
			// mutex resta acquisito per sempre e l'altro processo si blocca.
		case WAIT_ABANDONED:
		case WAIT_OBJECT_0:
		{
			if (isQuitRequested()) {
				isRunning = false;
			}
			break;
		}
		case WAIT_TIMEOUT:
		{
			break;
		}
		}
	}

	for (auto& t : threads) {
		if (t.joinable()) {
			t.join();
		}
	}
	threads.clear();

	for (size_t i = 0; i < num_control_points; i++)
	{
		if (hControlPointMutex[i] != NULL && hControlPointMutex[i] != INVALID_HANDLE_VALUE) {
			CloseHandle(hControlPointMutex[i]);
			hControlPointMutex[i] = NULL;
		}

		if (hControlPointEvent[i] != NULL && hControlPointEvent[i] != INVALID_HANDLE_VALUE) {
			CloseHandle(hControlPointEvent[i]);
			hControlPointEvent[i] = NULL;
		}

		if (hControlPointResults[i] != NULL && hControlPointResults[i] != INVALID_HANDLE_VALUE) {
			CloseHandle(hControlPointResults[i]);
			hControlPointResults[i] = NULL;
		}
	}

	if (hListMutex != NULL && hListMutex != INVALID_HANDLE_VALUE) {
		CloseHandle(hListMutex);
		hListMutex = NULL;
	}

	if (hListEventTrigger != NULL && hListEventTrigger != INVALID_HANDLE_VALUE) {
		CloseHandle(hListEventTrigger);
		hListEventTrigger = NULL;
	}

	if (hListEventAck != NULL && hListEventAck != INVALID_HANDLE_VALUE) {
		CloseHandle(hListEventAck);
		hListEventAck = NULL;
	}


	if (!errorDetected) {
		// FINAL RUN SUMMARY
		unsigned long totalDrops = 0;
		unsigned long totalSent = 0;
		LOG("\n===================== RUN SUMMARY =====================\n");
		for (int i = 0; i < num_control_points; i++) {
			const WaitStats& s = g_stats[i];
			totalDrops += s.frameDrops;
			totalSent += s.framesSent;
			const double avgMs = s.acksReceived > 0
				? (s.totalWaitUs / 1000.0) / s.acksReceived : 0.0;
			// frame drops  = inferenza ancora PENDING al tick (colpa dell'AI)
			// missed ticks = risveglio del simulatore in ritardo >= 1 intervallo (colpa nostra)
			LOG("CP {} | frames sent: {:6} | results: {:6} | frame drops: {} | missed ticks (sim late): {}\n"
				"CP {} | ack wait (trigger -> RESULT_READY) min: {:.3f} ms | avg: {:.3f} ms | max: {:.3f} ms\n",
				i, s.framesSent, s.acksReceived, s.frameDrops, s.missedTicks,
				i, s.minWaitUs / 1000.0, avgMs, s.maxWaitUs / 1000.0);
		}
		LOGC(totalDrops > 0 ? fmt::color::red : fmt::color::green, /*stderr*/ false,
			"TOTAL FRAME DROPS: {} (out of {} frames sent)\n", totalDrops, totalSent);
		LOG("=======================================================\n");
	}

	// Smaltisce coda e ferma il logger PRIMA di ripristinare il timer e uscire,
	// cosi' il summary viene effettivamente stampato prima della terminazione.
	logger().stop();

	// Ripristina la risoluzione del timer di sistema.
	timeEndPeriod(1);
}

// Per eseguire il programma: CTRL+F5 oppure Debug > Avvia senza eseguire debug
// Per eseguire il debug del programma: F5 oppure Debug > Avvia debug

// Suggerimenti per iniziare: 
//   1. Usare la finestra Esplora soluzioni per aggiungere/gestire i file
//   2. Usare la finestra Team Explorer per connettersi al controllo del codice sorgente
//   3. Usare la finestra di output per visualizzare l'output di compilazione e altri messaggi
//   4. Usare la finestra Elenco errori per visualizzare gli errori
//   5. Passare a Progetto > Aggiungi nuovo elemento per creare nuovi file di codice oppure a Progetto > Aggiungi elemento esistente per aggiungere file di codice esistenti al progetto
//   6. Per aprire di nuovo questo progetto in futuro, passare a File > Apri > Progetto e selezionare il file con estensione sln