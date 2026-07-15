// ConsoleApplicationTestAI.cpp : Questo file contiene la funzione 'main', in cui inizia e termina l'esecuzione del programma.
//

#include "Windows.h"
#include "conio.h"
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <tchar.h>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <cwctype>
#include <fmt/core.h>
#include <fmt/color.h>

#define NUM_CONTROL_POINTS 3

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

// Runtime configuration (overridable via command line, see wmain)
std::filesystem::path g_inputFolder = L"D:\\emanuele\\Code\\ConsoleApplicationTestAI\\dataset";
std::wstring g_modelPath = L"D:\\emanuele\\Code\\ConsoleApplicationTestAI\\model\\yolo_pure.onnx";
InferenceType g_inferenceType = InferenceType::CLASSIFICATION;

HANDLE hcpListMMF = NULL;
PTcontrolPointsList cpListMMF = NULL;

std::thread threads[NUM_CONTROL_POINTS];
HANDLE hControlPointMutex[NUM_CONTROL_POINTS];
HANDLE hControlPointEvent[NUM_CONTROL_POINTS];
HANDLE hControlPointResults[NUM_CONTROL_POINTS];

HANDLE hListMutex = NULL;
HANDLE hListEventTrigger = NULL;
HANDLE hListEventAck = NULL;
int frameRate_milliseconds = 60;

// Per-control-point statistics, printed once the program terminates
std::atomic<unsigned long> g_frameDropCount[NUM_CONTROL_POINTS];
std::atomic<unsigned long> g_framesSent[NUM_CONTROL_POINTS];

// Convert wide strings coming from the IPC structs / command line so all
// logging can go through narrow fmt::print (avoids mixing stream orientations)
static std::string ToNarrow(const std::wstring& wide) {
	return std::filesystem::path(wide).string();
}

void controlPointThreadFunc(int i)
{
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

	// Load image paths from the input folder
	std::vector<std::string> imagePaths;

	try {
		for (const auto& entry : std::filesystem::recursive_directory_iterator(g_inputFolder)) {
			if (entry.is_regular_file()) {
				std::string ext = entry.path().extension().string();
				std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
				if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
					imagePaths.push_back(entry.path().string());
				}
			}
		}
	}
	catch (const std::exception& e) {
		fmt::print(stderr, "### ERROR reading directory: {}\n", e.what());
	}

	fmt::print("Thread {} running (Simulating Frame Grabber). Loaded {} images.\n", i, imagePaths.size());

	int imageIndex = 0;
	std::string lastSentFilename = "NONE";

	// Frame staging: decode + resize happen OUTSIDE the trigger critical path
	// (during the idle time between two triggers), so at trigger time the only
	// remaining cost is a memcpy into shared memory
	cv::Mat stagedFrame;
	std::string stagedFilename = "NONE";

	auto stageNextFrame = [&]() {
		stagedFrame.release();
		if (imagePaths.empty()) {
			return;
		}
		const std::string& currentImagePath = imagePaths[imageIndex];
		stagedFilename = std::filesystem::path(currentImagePath).filename().string();

		cv::Mat img = cv::imread(currentImagePath, cv::IMREAD_COLOR);
		if (img.empty()) {
			fmt::print(stderr, "### ERROR: Failed to decode {}\n", stagedFilename);
		}
		else {
			// Force resize to match the Shared Memory allocated layout
			cv::resize(img, stagedFrame, cv::Size(cpListMMF->points[i].sizeX, cpListMMF->points[i].sizeY));
		}

		// Advance the index and loop back if we hit the end of the folder
		imageIndex = (imageIndex + 1) % static_cast<int>(imagePaths.size());
	};

	// Copies the staged frame into shared memory and triggers the AI engine.
	// Must be called with hControlPointMutex[i] held.
	auto sendStagedFrame = [&]() {
		if (pImage != NULL) {
			if (!stagedFrame.empty() && stagedFrame.isContinuous()) {
				memcpy(pImage, stagedFrame.data, nPayload);
				lastSentFilename = stagedFilename;
			}
			else if (imagePaths.empty()) {
				// Fallback mechanism if the folder is empty or invalid
				uint8_t* pPixels = static_cast<uint8_t*>(pImage);
				for (int p = 0; p < nPayload; p++) {
					pPixels[p] = static_cast<uint8_t>(p % 2);
				}
			}
		}

		// TRIGGER AI ENGINE FOR THE NEW FRAME
		cpListMMF->points[i].results.state = InferenceState::PENDING;
		g_framesSent[i]++;
		ResetEvent(hControlPointResults[i]);
		SetEvent(hControlPointEvent[i]);
	};

	// Stage the first frame before entering the timed loop. The trigger cadence
	// is hardware-faithful from t0: a real camera cannot be slowed down, so no
	// result is awaited before starting. Cold-start mitigation belongs to the
	// AI engine warmup (done at Configure time, before the camera starts); if
	// the first inference still exceeds the timeframe, that is a genuine drop
	// and is reported and counted as such.
	stageNextFrame();

	// HARDWARE TRIGGER SIMULATION: strict interval from the start
	const auto hardwareTriggerInterval = std::chrono::milliseconds(frameRate_milliseconds);
	auto next_trigger_time = std::chrono::high_resolution_clock::now() + hardwareTriggerInterval;

	bool keepRunning = true;

	while (keepRunning)
	{
		std::this_thread::sleep_until(next_trigger_time);
		next_trigger_time += hardwareTriggerInterval;

		bool frameSent = false;
		DWORD dwWaitResult = WaitForSingleObject(hControlPointMutex[i], INFINITE);

		if (dwWaitResult == WAIT_OBJECT_0 || dwWaitResult == WAIT_ABANDONED)
		{
			if (cpListMMF->points[i].status == PointState::QUIT) {
				keepRunning = false;
			}
			else
			{
				if (cpListMMF->points[i].results.state == InferenceState::PENDING) {
					g_frameDropCount[i]++;
					fmt::print(stderr, fg(fmt::color::yellow),
						"### WARNING: FRAME DROP DETECTED ON CP {}! AI execution exceeded timeframe (drops so far: {}).\n",
						i, g_frameDropCount[i].load());
				}
				else
				{
					// READ RESULTS OF THE PREVIOUS FRAME
					if (cpListMMF->points[i].results.state == InferenceState::RESULT_READY) {

						// The JSON payload is plain ASCII: convert it once and parse narrow
						std::wstring jsonW(cpListMMF->points[i].results.json);
						const wchar_t* p = cpListMMF->points[i].results.json;
						std::string jsonStr;
						jsonStr.reserve(wcsnlen(p, 1024));
						for (size_t k = 0; k < 1024 && p[k]; ++k)
							jsonStr.push_back(static_cast<char>(p[k]));  // cast esplicito -> niente C4244

						if (cpListMMF->points[i].inferenceType == InferenceType::ANOMALY) {
							float anomalyScore = -1.0f;
							std::string status = "UNKNOWN";

							// nlohmann/json emits compact JSON: {"anomaly_score":0.123,"status":"OK"}
							const std::string scoreKey = "\"anomaly_score\":";
							size_t scorePos = jsonStr.find(scoreKey);
							if (scorePos != std::string::npos) {
								try { anomalyScore = std::stof(jsonStr.substr(scorePos + scoreKey.length())); }
								catch (const std::exception& e) {
									fmt::print(stderr, "Exception occurred: {}\n", e.what());
								}
							}

							const std::string statusKey = "\"status\":\"";
							size_t statusPos = jsonStr.find(statusKey);
							if (statusPos != std::string::npos) {
								size_t valStart = statusPos + statusKey.length();
								size_t valEnd = jsonStr.find('"', valStart);
								if (valEnd != std::string::npos) {
									status = jsonStr.substr(valStart, valEnd - valStart);
								}
							}

							fmt::print(">> CP {} | FILE: {} | ANOMALY_SCORE: {} | STATUS: {}\n",
								i, lastSentFilename, anomalyScore, status);
						}
						else if (cpListMMF->points[i].inferenceType == InferenceType::CLASSIFICATION) {
							float confidence = -1.0f;
							int class_id = -1;

							const std::string confKey = "\"confidence\":";
							size_t confPos = jsonStr.find(confKey);
							if (confPos != std::string::npos) {
								try { confidence = std::stof(jsonStr.substr(confPos + confKey.length())); }
								catch (const std::exception& e) {
									fmt::print(stderr, "Exception occurred: {}\n", e.what());
								}
							}

							const std::string classKey = "\"class_id\":";
							size_t classPos = jsonStr.find(classKey);
							if (classPos != std::string::npos) {
								try { class_id = std::stoi(jsonStr.substr(classPos + classKey.length())); }
								catch (const std::exception& e) {
									fmt::print(stderr, "Exception occurred: {}\n", e.what());
								}
							}

							// Print the result alongside the exact filename that generated it
							fmt::print(">> CP {} | FILE: {} | PRED_CLASS: {} | CONFIDENCE: {}\n",
								i, lastSentFilename, class_id, confidence);
						}
					}
					else if (cpListMMF->points[i].results.state == InferenceState::ERROR_DETECTED) {
						fmt::print(stderr, "### POINT {} AI ERROR DETECTED!\n", i);
					}

					// --- WRITE THE (PRE-DECODED) FRAME INTO SHARED MEMORY AND TRIGGER ---
					sendStagedFrame();
					frameSent = true;
				}
			}
			ReleaseMutex(hControlPointMutex[i]);
		}

		// Prepare the next frame during the idle time until the next trigger.
		// On a dropped tick the staged frame is kept for the next attempt.
		if (keepRunning && frameSent) {
			stageNextFrame();
		}
	}

	if (pImage) UnmapViewOfFile(pImage);
	if (hMMFImage) CloseHandle(hMMFImage);
	if (pResImage) UnmapViewOfFile(pResImage);
	if (hMMFResImage) CloseHandle(hMMFResImage);

	fmt::print("Thread {} stopped! (frames sent: {}, frame drops: {})\n",
		i, g_framesSent[i].load(), g_frameDropCount[i].load());
}

void Quit() {
	DWORD mutexWait = WaitForSingleObject(hListMutex, INFINITE);
	switch (mutexWait) {
	case WAIT_OBJECT_0:
	{
		__try {
			cpListMMF->state = ListState::QUIT;
			for (int i = 0; i < NUM_CONTROL_POINTS; i++) {
				DWORD innerMutexWait = WaitForSingleObject(hControlPointMutex[i], INFINITE);
				if (innerMutexWait == WAIT_OBJECT_0) {
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
			fmt::print("ONNX manager shutdown acknowledged.\n");
		}
		else {
			fmt::print("### ONNX manager shutdown timeout!\n");
		}
		break;
	}
	case WAIT_TIMEOUT:
	{
		fmt::print("Main thread waiting for the global mutex!\n");
		break;
	}
	}
}

void Start()
{
	static bool isStarted = false;
	if (isStarted) {
		fmt::print("### WARNING: Simulation already running!\n");
		return;
	}
	isStarted = true;
	for (int i = 0; i < NUM_CONTROL_POINTS; i++)
	{
		threads[i] = std::thread(controlPointThreadFunc, i);
		//std::this_thread::sleep_for(std::chrono::milliseconds(30));
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
		fmt::print(stderr, "CreateMutex error: {}\n", GetLastError());
	}
	else {
		fmt::print("CreateMutex {}: {}\n",
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
		fmt::print(stderr, "CreateEvent error: {}\n", GetLastError());
	}
	else {
		fmt::print("CreateEvent {}: {}\n",
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
		cpListMMF->numPunti = NUM_CONTROL_POINTS;
		DWORD nRand = rand();

		for (int i = 0; i < NUM_CONTROL_POINTS; i++)
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

		fmt::print("Release the global mutex\n");
		if (!ReleaseMutex(hListMutex))
		{
			fmt::print(stderr, "### Release global mutex error\n");
		}

		// Notify the AI to start configuration (and TensorRT warmup)
		SetEvent(hListEventTrigger);

		fmt::print(">> Waiting for AI engine configuration...\n");
		fmt::print(">> [NOTE: First-time TensorRT optimization takes time. Press 'q' to abort]\n");

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
			fmt::print("### WARNING: Configuration interrupted by user. Initiating shutdown...\n");
			Quit();
			return false;
		}

		// Update the global state
		WaitForSingleObject(hListMutex, INFINITE);
		if (cpListMMF->state == ListState::CONFIGURED) {
			fmt::print("Configuration control points completed!\n");
			isConfigured = true;
		}
		else {
			fmt::print(stderr, "### ERROR: control points configuration failed.\n");
		}
		ReleaseMutex(hListMutex);

		break;
	}
	case WAIT_TIMEOUT:
	{
		fmt::print("Main thread waiting for the global mutex!\n");
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
	fmt::print("DELTA VISIONE - Test AI via MMF + ONNX!\n");

	// Optional positional arguments; defaults defined at the top of the file
	// Usage: ConsoleApplicationTestAI.exe [inputFolder] [modelPath] [anomaly|classification|object_detection]
	if (argc > 1) {
		g_inputFolder = argv[1];
	}
	if (argc > 2) {
		g_modelPath = argv[2];
		if (g_modelPath.length() >= 512) {
			fmt::print(stderr, "### ERROR: model path exceeds 511 characters.\n");
			return 1;
		}
	}
	if (argc > 3) {
		if (!parseInferenceType(argv[3], g_inferenceType)) {
			fmt::print(stderr, "### ERROR: unknown analysis type '{}'. Valid values: anomaly (0), classification (1), object_detection (2).\n",
				ToNarrow(argv[3]));
			return 1;
		}
	}

	if (argc > 4) {
		try {
			frameRate_milliseconds = std::stoi(argv[4]);
		}
		catch (const std::exception&) {
			fmt::print(stderr, "### ERROR: invalid frame rate value: '{}'. Must be an integer.\n", ToNarrow(argv[4]));
			return 1;
		}
	}

	if (!std::filesystem::is_directory(g_inputFolder)) {
		fmt::print(stderr, "### WARNING: input folder does not exist or is not a directory: {}\n", g_inputFolder.string());
	}
	if (!std::filesystem::is_regular_file(g_modelPath)) {
		fmt::print(stderr, "### WARNING: model file not found: {}\n", ToNarrow(g_modelPath));
	}

	fmt::print(">> Input folder : {}\n"
		">> Model path   : {}\n"
		">> Analysis type: {}\n"
		">> Trigger rate : {} ms\n\n",
		g_inputFolder.string(),
		ToNarrow(g_modelPath),
		g_inferenceType == InferenceType::ANOMALY ? "ANOMALY" :
		g_inferenceType == InferenceType::CLASSIFICATION ? "CLASSIFICATION" : "OBJECT_DETECTION",
		frameRate_milliseconds);

	srand(1792);

	hListMutex = CreateMutex(NULL, FALSE, TEXT("LISTMUTEX"));
	hListEventTrigger = CreateEvent(NULL, FALSE, FALSE, TEXT("LISTEVENTTRIGGER"));
	hListEventAck = CreateEvent(NULL, FALSE, FALSE, TEXT("LISTEVENTACK"));


	ZeroMemory(hControlPointMutex, sizeof(HANDLE) * NUM_CONTROL_POINTS);
	ZeroMemory(hControlPointEvent, sizeof(HANDLE) * NUM_CONTROL_POINTS);
	ZeroMemory(hControlPointResults, sizeof(HANDLE) * NUM_CONTROL_POINTS);

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
		fmt::print(stderr, "Could not map view of file ({}).\n", GetLastError());
		CloseHandle(hcpListMMF);
		return 1;
	}

	fmt::print("Press 'q' to quit, press 'c' to configure, press 's' to start sim\n");

	bool isRunning = true;
	bool isConfigured = false;
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
					fmt::print(stderr, "### ERROR: Configuration failed...starting shutdown!\n");
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
					fmt::print(stderr, "### ERROR: Configure before starting!\n");
				}
				break;
			}

			case 'q':
				fmt::print("Initiating shutdown sequence...\n");
				Quit();
				break;

			default:
				// Ignore any other key presses
				break;
			}
		}
		DWORD resWait = WaitForSingleObject(hListMutex, 100);
		switch (resWait) {
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

	for (size_t i = 0; i < NUM_CONTROL_POINTS; i++)
	{
		if (threads[i].joinable()) {
			threads[i].join();
		}
		CloseHandle(hControlPointMutex[i]);
		CloseHandle(hControlPointEvent[i]);
		CloseHandle(hControlPointResults[i]);
	}

	CloseHandle(hListMutex);
	CloseHandle(hListEventTrigger);
	CloseHandle(hListEventAck);

	// --- FINAL RUN SUMMARY ---
	unsigned long totalDrops = 0;
	unsigned long totalSent = 0;
	fmt::print("\n============ RUN SUMMARY ============\n");
	for (int i = 0; i < NUM_CONTROL_POINTS; i++) {
		const unsigned long drops = g_frameDropCount[i].load();
		const unsigned long sent = g_framesSent[i].load();
		totalDrops += drops;
		totalSent += sent;
		fmt::print("CP {} | frames sent: {:6} | frame drops: {}\n", i, sent, drops);
	}
	fmt::print(totalDrops > 0 ? fg(fmt::color::red) : fg(fmt::color::green),
		"TOTAL FRAME DROPS: {} (out of {} frames sent)\n", totalDrops, totalSent);
	fmt::print("=====================================\n");
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