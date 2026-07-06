// ConsoleApplicationTestAI.cpp : Questo file contiene la funzione 'main', in cui inizia e termina l'esecuzione del programma.
//

#include "Windows.h"
#include "conio.h"
#include <chrono>
#include <iostream>
#include <thread>
#include <tchar.h>
#include <string>

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

	// Create dynamic wide string using standard C++
	std::wstring pName = L"MMF_" + std::to_wstring(cpListMMF->points[i].idPunto) + L"_IMAGE";

	HANDLE hMMFImage = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, nPayload, pName.c_str());
	void* pImage = NULL;
	if (hMMFImage != NULL) {
		pImage = MapViewOfFile(hMMFImage, FILE_MAP_WRITE, 0, 0, nPayload);
	}

	// Map OUTPUT RESIMAGE (AI -> Sender)
	int nResPayload = cpListMMF->points[i].sizeX * cpListMMF->points[i].sizeY;

	// Create dynamic wide string
	std::wstring resName = L"MMF_" + std::to_wstring(cpListMMF->points[i].idPunto) + L"_RESIMAGE";

	HANDLE hMMFResImage = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, nResPayload, resName.c_str());
	void* pResImage = NULL;
	if (hMMFResImage != NULL) {
		pResImage = MapViewOfFile(hMMFResImage, FILE_MAP_READ, 0, 0, nResPayload);
	}

	std::cout << "Thread " << i << " running\n";

	// Simulated camera trigger
	const int CAMERA_FPS = 25;
	const auto cameraInterval = std::chrono::milliseconds(1000 / CAMERA_FPS);

	while (cpListMMF->points[i].status != PointState::QUIT)
	{
		auto loop_start = std::chrono::high_resolution_clock::now();

		DWORD dwWaitResult = WaitForSingleObject(hControlPointMutex[i], INFINITE);

		switch (dwWaitResult)
		{
		case WAIT_OBJECT_0:
		{
			// Frame dropping
			// If state is PENDING, the AI has not finished processing the previous frame.
			if (cpListMMF->points[i].results.state == InferenceState::PENDING) {
				std::cerr << "### WARNING: FRAME DROP DETECTED ON CP " << i
					<< "! AI is too slow for " << CAMERA_FPS << " FPS.\n";
			}
			else
			{
				// If available, read preavious result
				if (cpListMMF->points[i].results.state == InferenceState::RESULT_READY) {

					std::wstring jsonW(cpListMMF->points[i].results.json);
					float anomalyScore = -1.0f;
					std::wstring status = L"UNKNOWN";

					// Parse anomaly score
					std::wstring scoreKey = L"\"anomaly_score\":";
					size_t scorePos = jsonW.find(scoreKey);
					if (scorePos != std::wstring::npos) {
						try {
							anomalyScore = std::stof(jsonW.substr(scorePos + scoreKey.length()));
						}
						catch (const std::invalid_argument& e) {
							std::wcerr << L"### ERROR CP " << i << L": anomaly_score non è un numero valido (JSON malformato)!\n";
						}
						catch (const std::out_of_range& e) {
							std::wcerr << L"### ERROR CP " << i << L": anomaly_score fuori range per un float!\n";
						}
					}

					// Parse status
					std::wstring statusKey = L"\"status\": \"";
					size_t statusPos = jsonW.find(statusKey);
					if (statusPos != std::wstring::npos) {
						size_t valStart = statusPos + statusKey.length();
						size_t valEnd = jsonW.find(L"\"", valStart);
						if (valEnd != std::wstring::npos) {
							status = jsonW.substr(valStart, valEnd - valStart);
						}
					}

					std::wcout << L">> POINT " << i << L" AI RESULT -> Score: "
						<< anomalyScore << L" | Status: " << status << L"\n";

					// Set state back to IDLE to acknowledge data reception
					cpListMMF->points[i].results.state = InferenceState::IDLE;
				}
				else if (cpListMMF->points[i].results.state == InferenceState::ERROR_DETECTED) {
					std::wcerr << L"### POINT " << i << L" AI ERROR DETECTED!\n";
				}

				// Generate new frame
				if (pImage != NULL) {
					uint8_t* pPixels = static_cast<uint8_t*>(pImage);
					for (int p = 0; p < nPayload; p++) {
						pPixels[p] = static_cast<uint8_t>(p % 2); // Simulated test pattern
					}
				}

				// trigger AI
				cpListMMF->points[i].results.state = InferenceState::PENDING;
				ResetEvent(hControlPointResults[i]);
				SetEvent(hControlPointEvent[i]);
			}

			// Always safely release the mutex so the AI can lock it
			if (!ReleaseMutex(hControlPointMutex[i])) {
				std::cerr << "### Thread " << i << " release mutex error\n";
			}
			break;
		}
		case WAIT_TIMEOUT:
			std::cout << "@@@ Thread " << i << " is waiting for mutex...\n";
			break;
		case WAIT_ABANDONED:
			break;
		}

		// HARDWARE TRIGGER SIMULATION (THROTTLING)
		// Calculate how long this iteration took and sleep the remaining time 
		// to strictly enforce the target FPS.
		auto loop_end = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(loop_end - loop_start);

		if (elapsed < cameraInterval) {
			std::this_thread::sleep_for(cameraInterval - elapsed);
		}
	}

	// Resource cleanup to prevent severe memory leaks
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
	case WAIT_ABANDONED:
	{
		break;
	}
	}
}

void Start()
{
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

void Configure()
{
	DWORD globalWaitResult = WaitForSingleObject(hListMutex, INFINITE);

	switch (globalWaitResult)
	{
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

			// Specific path for your EfficientAD model
			wcscpy_s(cpListMMF->points[i].pathModello, 512, TEXT("D:/emanuele/Code/anomaly_detection_for_textile_industry/exports/efficientad_onnx/weights/onnx/26062026_083600_dustOnValidationAndTrain_EfficientAd_wide_resnet50_2.onnx"));

			swprintf_s(cpListMMF->points[i].mutexName, 128, TEXT("CP_%lu_MUTEX"), cpListMMF->points[i].idPunto);
			swprintf_s(cpListMMF->points[i].eventReadyName, 128, TEXT("CP_%lu_EVENTREADY"), cpListMMF->points[i].idPunto);
			swprintf_s(cpListMMF->points[i].resultsEventName, 128, TEXT("CP_%lu_EVENTRESULTS"), cpListMMF->points[i].idPunto);

			createMutex(hControlPointMutex[i], i);
			createEvent(hControlPointEvent[i], i, L"EVENTREADY");
			createEvent(hControlPointResults[i], i, L"EVENTRESULTS");

			cpListMMF->points[i].inferenceType = InferenceType::ANOMALY;
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
			return;
		}

		// Verify that each point is correctly initialized
		int correctlyConfigured = 0;
		for (int j = 0; j < NUM_CONTROL_POINTS; j++) {
			WaitForSingleObject(hControlPointMutex[j], INFINITE);
			if (cpListMMF->points[j].status != PointState::CONFIGURED) {
				correctlyConfigured = 1;
			}
			ReleaseMutex(hControlPointMutex[j]);
		}

		// Update the global state
		WaitForSingleObject(hListMutex, INFINITE);
		if (correctlyConfigured == 0) {
			cpListMMF->state = ListState::CONFIGURED;
			std::cout << "Configuration control points completed!" << std::endl;
		}
		else {
			cpListMMF->state = ListState::ERROR_DETECTED;
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
	case WAIT_ABANDONED:
	{
		break;
	}
	}
}

int main()
{
	std::wcout << TEXT("DELTA VISIONE - Test AI via MMF + ONNX!\n");

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
	while (isRunning)
	{
		if (_kbhit())
		{
			char c = _getch();

			switch (c)
			{
			case 'c':
			{
				Configure();
				bool isConfigured = false;
				WaitForSingleObject(hListMutex, INFINITE);
				if (cpListMMF->state == ListState::CONFIGURED) {
					isConfigured = true;
				}
				ReleaseMutex(hListMutex);
				if (!isConfigured) {
					std::cout << "### ERROR: Configuration failed before...starting shutdown!" << std::endl;
					Quit();
				}
				break;
			}
			case 's':
			{
				bool isConfigured = false;
				WaitForSingleObject(hListMutex, INFINITE);
				if (cpListMMF->state == ListState::CONFIGURED) {
					isConfigured = true;
				}
				ReleaseMutex(hListMutex);

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
			case WAIT_ABANDONED:
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
