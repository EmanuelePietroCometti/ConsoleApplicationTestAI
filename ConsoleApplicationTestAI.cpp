// ConsoleApplicationTestAI.cpp : Questo file contiene la funzione 'main', in cui inizia e termina l'esecuzione del programma.
//

#include "Windows.h"
#include "conio.h"
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
		std::cerr << "### ERROR reading directory: " << e.what() << "\n";
	}

	std::cout << "Thread " << i << " running (Simulating Frame Grabber). Loaded " << imagePaths.size() << " images.\n";

	int imageIndex = 0;
	std::string lastSentFilename = "NONE";

	// HARDWARE TRIGGER SIMULATION: Strict 100ms interval (10 FPS)
	const auto hardwareTriggerInterval = std::chrono::milliseconds(500);
	auto next_trigger_time = std::chrono::high_resolution_clock::now() + hardwareTriggerInterval;

	bool keepRunning = true;

	while (keepRunning)
	{
		std::this_thread::sleep_until(next_trigger_time);
		next_trigger_time += hardwareTriggerInterval;

		DWORD dwWaitResult = WaitForSingleObject(hControlPointMutex[i], INFINITE);

		if (dwWaitResult == WAIT_OBJECT_0 || dwWaitResult == WAIT_ABANDONED)
		{
			if (cpListMMF->points[i].status == PointState::QUIT) {
				keepRunning = false;
			}
			else
			{
				if (cpListMMF->points[i].results.state == InferenceState::PENDING) {
					std::cerr << "### WARNING: FRAME DROP DETECTED ON CP " << i << "! AI execution exceeded timeframe.\n";
				}
				else
				{
					// READ RESULTS OF THE PREVIOUS FRAME
					if (cpListMMF->points[i].results.state == InferenceState::RESULT_READY) {

						std::wstring jsonW(cpListMMF->points[i].results.json);

						if (cpListMMF->points[i].inferenceType == InferenceType::ANOMALY) {
							float anomalyScore = -1.0f;
							std::wstring status = L"UNKNOWN";

							// nlohmann/json emits compact JSON: {"anomaly_score":0.123,"status":"OK"}
							std::wstring scoreKey = L"\"anomaly_score\":";
							size_t scorePos = jsonW.find(scoreKey);
							if (scorePos != std::wstring::npos) {
								try { anomalyScore = std::stof(jsonW.substr(scorePos + scoreKey.length())); }
								catch (...) {}
							}

							std::wstring statusKey = L"\"status\":\"";
							size_t statusPos = jsonW.find(statusKey);
							if (statusPos != std::wstring::npos) {
								size_t valStart = statusPos + statusKey.length();
								size_t valEnd = jsonW.find(L"\"", valStart);
								if (valEnd != std::wstring::npos) {
									status = jsonW.substr(valStart, valEnd - valStart);
								}
							}

							std::wcout << L">> CP " << i << L" | FILE: "
								<< std::filesystem::path(lastSentFilename).wstring()
								<< L" | ANOMALY_SCORE: " << anomalyScore << L" | STATUS: " << status << L"\n";
						}
						else if (cpListMMF->points[i].inferenceType == InferenceType::CLASSIFICATION) {
							float confidence = -1.0f;
							int class_id = -1;

							std::wstring confKey = L"\"confidence\":";
							size_t confPos = jsonW.find(confKey);
							if (confPos != std::wstring::npos) {
								try { confidence = std::stof(jsonW.substr(confPos + confKey.length())); }
								catch (std::exception& e) {
									std::cerr << "Excpetion occurred: " << e.what() << std::endl;
								}
							}

							std::wstring classKey = L"\"class_id\":";
							size_t classPos = jsonW.find(classKey);
							if (classPos != std::wstring::npos) {
								try { class_id = std::stoi(jsonW.substr(classPos + classKey.length())); }
								catch (std::exception& e) {
									std::cerr << "Excpetion occurred: " << e.what() << std::endl;
								}
							}

							// Print the result alongside the exact filename that generated it
							std::cout << ">> CP " << i << " | FILE: " << lastSentFilename
								<< " | PRED_CLASS: " << class_id << " | CONFIDENCE: " << confidence << "\n";
						}
						else if (cpListMMF->points[i].inferenceType == InferenceType::ANOMALY) {
							float anomaly_score = -1.0f;
							bool is_anomaly = false;

							std::wstring scoreKey = L"\"anomaly_score\":";
							size_t scorePos = jsonW.find(scoreKey);
							if (scorePos != std::wstring::npos) {
								try { anomaly_score = std::stof(jsonW.substr(scorePos + scoreKey.length())); }
								catch (std::exception& e) {
									std::cerr << "Excpetion occurred: " << e.what() << std::endl;
								}
							}

							std::wstring isAnomalyKey = L"\"status\":";
							size_t isAnomalyPos = jsonW.find(isAnomalyKey);
							if (isAnomalyPos != std::wstring::npos) {
								try {
									std::wstring val = jsonW.substr(isAnomalyPos + isAnomalyKey.length());
									// Check if the JSON value contains "true" or "1"
									is_anomaly = (val.find(L"true") != std::wstring::npos || val.find(L"1") != std::wstring::npos);
								}
								catch (std::exception& e) {
									std::cerr << "Excpetion occurred: " << e.what() << std::endl;
								}
							}

							std::cout << ">> CP " << i << " | FILE: " << lastSentFilename
								<< " | ANOMALY SCORE: " << anomaly_score << " | IS_ANOMALY: " << is_anomaly << "\n";
						}
					}
					else if (cpListMMF->points[i].results.state == InferenceState::ERROR_DETECTED) {
						std::cerr << "### POINT " << i << " AI ERROR DETECTED!\n";
					}

					// --- WRITE THE NEW FRAME INTO SHARED MEMORY ---
					if (pImage != NULL) {
						if (!imagePaths.empty()) {
							std::string currentImagePath = imagePaths[imageIndex];

							// Extract filename for the next iteration's logging
							lastSentFilename = std::filesystem::path(currentImagePath).filename().string();

							// Read and decode the image via OpenCV
							cv::Mat img = cv::imread(currentImagePath, cv::IMREAD_COLOR);
							if (!img.empty()) {
								cv::Mat resizedImg;
								// Force resize to match the Shared Memory allocated layout
								cv::resize(img, resizedImg, cv::Size(cpListMMF->points[i].sizeX, cpListMMF->points[i].sizeY));

								// Memcopy the contiguous raw BGR bytes into the MMF
								if (resizedImg.isContinuous()) {
									memcpy(pImage, resizedImg.data, nPayload);
								}
							}
							else {
								std::cerr << "### ERROR: Failed to decode " << lastSentFilename << "\n";
							}

							// Advance the index and loop back if we hit the end of the folder
							imageIndex = (imageIndex + 1) % imagePaths.size();
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
					ResetEvent(hControlPointResults[i]);
					SetEvent(hControlPointEvent[i]);
				}
			}
			ReleaseMutex(hControlPointMutex[i]);
		}
	}

	if (pImage) UnmapViewOfFile(pImage);
	if (hMMFImage) CloseHandle(hMMFImage);
	if (pResImage) UnmapViewOfFile(pResImage);
	if (hMMFResImage) CloseHandle(hMMFResImage);

	std::cout << "Thread " << i << " stopped!\n";
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
			std::cout << "ONNX manager shutdown acknowledged." << std::endl;
		}
		else {
			std::cout << "### ONNX manager shutdown timeout!" << std::endl;
		}
		break;
	}
	case WAIT_TIMEOUT:
	{
		std::cout << "Main thread waiting for the global mutex!" << std::endl;
		break;
	}
	}
}

void Start()
{
	static bool isStarted = false;
	if (isStarted) {
		std::cout << "### WARNING: Simulation already running!\n";
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
		std::wcout << TEXT("CreateMutex error: ") << GetLastError() << std::endl;
	}
	else {
		std::wcout << TEXT("CreateMutex ") << ((GetLastError() == ERROR_ALREADY_EXISTS) ? TEXT("opened existing") : TEXT("created new")) << TEXT(": ") << name << std::endl;
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
		std::wcout << TEXT("CreateEvent error: ") << GetLastError() << std::endl;
	}
	else {
		std::wcout << TEXT("CreateEvent ") << ((GetLastError() == ERROR_ALREADY_EXISTS) ? TEXT("opened existing") : TEXT("created new")) << TEXT(": ") << name << std::endl;
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

		std::cout << "Release the global mutex " << std::endl;
		if (!ReleaseMutex(hListMutex))
		{
			std::cout << "### Release global mutex error" << std::endl;
		}

		// Notify the AI to start configuration (and TensorRT warmup)
		SetEvent(hListEventTrigger);

		std::wcout << L">> Waiting for AI engine configuration..." << std::endl;
		std::wcout << L">> [NOTE: First-time TensorRT optimization takes time. Press 'q' to abort]" << std::endl;

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
			std::wcout << L"### WARNING: Configuration interrupted by user. Initiating shutdown..." << std::endl;
			Quit();
			return false;
		}

		// Update the global state
		WaitForSingleObject(hListMutex, INFINITE);
		if (cpListMMF->state == ListState::CONFIGURED) {
			std::cout << "Configuration control points completed!" << std::endl;
			isConfigured = true;
		}
		else {
			std::cout << "### ERROR: control points configuration failed." << std::endl;
		}
		ReleaseMutex(hListMutex);

		break;
	}
	case WAIT_TIMEOUT:
	{
		std::cout << "Main thread waiting for the global mutex!" << std::endl;
		break;
	}
	}
	return isConfigured;
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
	std::wcout << TEXT("DELTA VISIONE - Test AI via MMF + ONNX!\n");

	// Optional positional arguments; defaults defined at the top of the file
	// Usage: ConsoleApplicationTestAI.exe [inputFolder] [modelPath] [anomaly|classification|object_detection]
	if (argc > 1) {
		g_inputFolder = argv[1];
	}
	if (argc > 2) {
		g_modelPath = argv[2];
		if (g_modelPath.length() >= 512) {
			std::wcerr << L"### ERROR: model path exceeds 511 characters." << std::endl;
			return 1;
		}
	}
	if (argc > 3) {
		if (!parseInferenceType(argv[3], g_inferenceType)) {
			std::wcerr << L"### ERROR: unknown analysis type '" << argv[3]
				<< L"'. Valid values: anomaly (0), classification (1), object_detection (2)." << std::endl;
			return 1;
		}
	}

	if (!std::filesystem::is_directory(g_inputFolder)) {
		std::wcerr << L"### WARNING: input folder does not exist or is not a directory: "
			<< g_inputFolder.wstring() << std::endl;
	}
	if (!std::filesystem::is_regular_file(g_modelPath)) {
		std::wcerr << L"### WARNING: model file not found: " << g_modelPath << std::endl;
	}

	std::wcout << L">> Input folder : " << g_inputFolder.wstring() << L"\n"
		<< L">> Model path   : " << g_modelPath << L"\n"
		<< L">> Analysis type: "
		<< (g_inferenceType == InferenceType::ANOMALY ? L"ANOMALY" :
			g_inferenceType == InferenceType::CLASSIFICATION ? L"CLASSIFICATION" : L"OBJECT_DETECTION")
		<< L"\n" << std::endl;

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
		std::cout << "Could not map view of file (" << GetLastError() << ")." << std::endl;
		CloseHandle(hcpListMMF);
		return 1;
	}

	std::cout << "Press 'q' to quit, press 'c' to configure, press 's' to start sim" << std::endl;

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
					std::cout << "### ERROR: Configuration failed...starting shutdown!" << std::endl;
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
					std::cout << "### ERROR: Configure before starting!" << std::endl;
				}
				break;
			}

			case 'q':
				std::cout << "Initiating shutdown sequence..." << std::endl;
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
				__try {
					if (cpListMMF->state == ListState::QUIT) {
						isRunning = false;
					}
				}
				__finally {
					ReleaseMutex(hListMutex);
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
