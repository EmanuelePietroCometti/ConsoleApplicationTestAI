// ConsoleApplicationTestAI.cpp : Questo file contiene la funzione 'main', in cui inizia e termina l'esecuzione del programma.
//

#include "Windows.h"
#include "conio.h"
#include <chrono>
#include <iostream>
#include <thread>
#include <tchar.h>

#define NUM_CONTROL_POINTS 3

enum class InferenceState : WORD {
	IDLE = 0,
	PENDING = 1,
	RESULT_READY = 2,
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
	InferenceType inferenceType;
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
	int nPayload = cpListMMF->points[i].sizeX * cpListMMF->points[i].sizeY * cpListMMF->points[i].bpp / 8;

	TCHAR pName[64];
	swprintf_s(pName, 64, TEXT("MMF_%lu_IMAGE"), cpListMMF->points[i].idPunto);
	HANDLE hMMFImage = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, nPayload, pName);
	void* pImage = NULL;
	DWORD dwErr = GetLastError();
	if (dwErr == ERROR_SUCCESS || dwErr == ERROR_ALREADY_EXISTS)
		pImage = MapViewOfFile(hMMFImage, FILE_MAP_WRITE, 0, 0, nPayload);

	std::cout << "Thread " << i << "th running" << std::endl;

	while (cpListMMF->points[i].status != PointState::QUIT)
	{
		DWORD dwWaitResult = WaitForSingleObject(
			hControlPointMutex[i],    // handle to mutex
			1);  // no time-out interval

		switch (dwWaitResult)
		{
			// The thread got ownership of the mutex
		case WAIT_OBJECT_0:
		{
			__try {
				auto start = std::chrono::high_resolution_clock::now();

				// Write simulated image data directly into the MMF
				if (pImage != NULL) {
					int nVal = rand() % 256;
					memset(pImage, nVal, nPayload);
				}
				cpListMMF->points[i].results.state = InferenceState::PENDING;
				auto end = std::chrono::high_resolution_clock::now();

				auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

				std::cout << "Thread " << i << "th set image and event - time " << duration.count() << " us" << std::endl;

			}

			__finally {
				// Release ownership of the mutex object
				std::cout << "Thread " << i << "th release mutex" << std::endl;
				if (!ReleaseMutex(hControlPointMutex[i]))
				{
					std::cout << "### Thread " << i << "th release mutex error" << std::endl;
				}
			}
			auto start = std::chrono::high_resolution_clock::now();
			SetEvent(hControlPointEvent[i]);

			DWORD resWait = WaitForSingleObject(hControlPointResults[i], 2000);

			if (resWait == WAIT_OBJECT_0) {


				DWORD mutexWait = WaitForSingleObject(hControlPointMutex[i], INFINITE);
				if (mutexWait == WAIT_OBJECT_0) {
					__try {
						// Verify that the inference finished successfully
						if (cpListMMF->points[i].results.state == InferenceState::RESULT_READY) {
							auto end = std::chrono::high_resolution_clock::now();
							auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

							// Process the JSON output 
							std::wcout << TEXT(">> POINT ") << i << TEXT(" AI RESULT [")
								<< duration.count() << TEXT(" ms]: ")
								<< cpListMMF->points[i].results.json << std::endl;

							// Acknowledge reading by setting state back to IDLE
							cpListMMF->points[i].results.state = InferenceState::IDLE;
						}
						// Handle potential errors reported by the TensorRT engine
						else if (cpListMMF->points[i].results.state == InferenceState::ERROR_DETECTED) {
							std::wcout << TEXT("### POINT ") << i << TEXT(" AI ERROR DETECTED!") << std::endl;
						}
					}
					__finally {
						ReleaseMutex(hControlPointMutex[i]);
					}
				}
			}
			else if (resWait == WAIT_TIMEOUT) {
				std::wcout << TEXT("### POINT ") << i << TEXT(" AI TIMEOUT ERROR!") << std::endl;
			}
			break;

			// The thread got ownership of an abandoned mutex
			// The database is in an indeterminate state
		}
		case WAIT_TIMEOUT:
		{
			std::cout << "@@@ Thread " << i << "is waiting for mutex..." << std::endl;
			break;
			// The thread got ownership of an abandoned mutex
					// The database is in an indeterminate state
		}
		case WAIT_ABANDONED:
			break;

		}

		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
	if (pImage) UnmapViewOfFile(pImage);
	if (hMMFImage) CloseHandle(hMMFImage);

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
	DWORD globalWaitResult = WaitForSingleObject(
		hListMutex,		// handle to mutex
		INFINITE);		// no time-out interval

	switch (globalWaitResult)
	{
		// The thread got ownership of the mutex
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
			wcscpy_s(cpListMMF->points[i].pathModello, 512, TEXT("D:\\modello.onnx"));

			swprintf_s(cpListMMF->points[i].mutexName, 128, TEXT("CP_%lu_MUTEX"), cpListMMF->points[i].idPunto);
			swprintf_s(cpListMMF->points[i].eventReadyName, 128, TEXT("CP_%lu_EVENTREADY"), cpListMMF->points[i].idPunto);
			swprintf_s(cpListMMF->points[i].resultsEventName, 128, TEXT("CP_%lu_EVENTRESULTS"), cpListMMF->points[i].idPunto);

			createMutex(hControlPointMutex[i], i);
			createEvent(hControlPointEvent[i], i, L"EVENTREADY");
			createEvent(hControlPointResults[i], i, L"EVENTRESULTS");

			cpListMMF->points[i].status = PointState::UPDATE_PENDING;
		}

		// Release ownership of the mutex object manually without __finally
		std::cout << "Release the global mutex " << std::endl;
		if (!ReleaseMutex(hListMutex))
		{
			std::cout << "### Release global mutex error" << std::endl;
		}
		SetEvent(hListEventTrigger);

		DWORD ackWait = WaitForSingleObject(hListEventAck, 3000);
		if (ackWait == WAIT_TIMEOUT) {
			std::wcout << "### ERROR: ONNX manager not reachable!" << std::endl;
			cpListMMF->state = ListState::IDLE;
			ReleaseMutex(hListMutex);
			break;
		}

		int correctlyConfigured = 0;
		for (int j = 0; j < NUM_CONTROL_POINTS; j++) {
			WaitForSingleObject(hControlPointMutex[j], INFINITE);
			if (cpListMMF->points[j].status != PointState::CONFIGURED) {
				correctlyConfigured = 1;
			}
			ReleaseMutex(hControlPointMutex[j]);
		}

		if (correctlyConfigured == 0) {
			WaitForSingleObject(hListMutex, INFINITE);
			cpListMMF->state = ListState::CONFIGURED;
			ReleaseMutex(hListMutex);
			std::cout << "Configuration control points completed!" << std::endl;
		}
		else {
			WaitForSingleObject(hListMutex, INFINITE);
			cpListMMF->state = ListState::ERROR_DETECTED;
			ReleaseMutex(hListMutex);
			std::cout << "### ERROR: control points configuration failed." << std::endl;
		}
		break;

		// The thread got ownership of an abandoned mutex
		// The database is in an indeterminate state
	}
	case WAIT_TIMEOUT:
	{
		std::cout << "Main thread waiting for the global mutex!" << std::endl;
		break;
		// The thread got ownership of an abandoned mutex
		// The database is in an indeterminate state
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
		char c = _getch();
		switch (c)
		{
		case 'c':
		{
			Configure();
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
			if (isConfigured)
				Start();
			else
				std::cout << "### Configure before start" << std::endl;
			break;
		}
		
		case 'q':
		{
			std::cout << "Initiating shutdwon sequence..." << std::endl;
			Quit();
			break;
		}
		
		default:
		{
			break;
		}
		}
		DWORD resWait = WaitForSingleObject(hListMutex, 10);
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
