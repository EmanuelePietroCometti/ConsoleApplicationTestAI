// ConsoleApplicationTestAI.cpp : Questo file contiene la funzione 'main', in cui inizia e termina l'esecuzione del programma.
//

#include "Windows.h"
#include "conio.h"
#include <chrono>
#include <iostream>
#include <thread>

#define NUM_CONTROL_POINTS 3

enum class InferenceState : WORD {
	PENDING = 0,
	RESULT_READY = 1,
	ERROR_DETECTED = 4
};
enum class PointState : WORD {
	IDLE = 0,
	CONFIGURED = 1,
	UPDATE_PENDING = 2,
	ERROR_DETECTED = 4
};
enum class ListState : WORD {
	CONFIGURED = 0,
	UPDATE_PENDING = 1
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
	TCHAR eventReadyName[128];
	PointState status;
	resultInference results;
} controlPoint, * PTcontrolPoint;

typedef struct controlPointsList {
	DWORD numPunti;
	ListState state;
	TCHAR listMutexName[128];
	TCHAR listEventAckName[128];
	controlPoint points[1024];
} controlPointsList, * PTcontrolPointsList;

HANDLE hcpListMMF = NULL;
PTcontrolPointsList cpListMMF = NULL;

std::thread threads[NUM_CONTROL_POINTS];
HANDLE hControlPointMutex[NUM_CONTROL_POINTS];
HANDLE hControlPointEvent[NUM_CONTROL_POINTS];

HANDLE hListMutex = NULL;
HANDLE hListEvent = NULL;

bool bQuit = false;
bool bCanStart = false;

void controlPointThreadFunc(int i)
{
	int nPayload = cpListMMF->points[i].sizeX * cpListMMF->points[i].sizeY * cpListMMF->points[i].bpp / 8;
	BYTE* pData = (BYTE*)malloc(nPayload);

	if (hControlPointMutex[i] != NULL)
		CloseHandle(hControlPointMutex[i]);
	hControlPointMutex[i] = NULL;

	hControlPointMutex[i] = CreateMutex(NULL, FALSE, cpListMMF->points[i].mutexName);
	if (hControlPointMutex[i] == NULL)
		printf("CreateMutex error: %d\n", GetLastError());
	else
		if (GetLastError() == ERROR_ALREADY_EXISTS)
			printf("CreateMutex opened an existing mutex\n");
		else printf("CreateMutex created a new mutex.\n");

	TCHAR pName[64];
	swprintf_s(pName, 64, TEXT("MMF_%u_IMAGE"), cpListMMF->points[i].idPunto);
	HANDLE hMMFImage = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, nPayload, pName);
	void* pImage = NULL;
	DWORD dwErr = GetLastError();
	if (dwErr == ERROR_SUCCESS || dwErr == ERROR_ALREADY_EXISTS)
		pImage = MapViewOfFile(hMMFImage, FILE_MAP_WRITE, 0, 0, nPayload);

	std::cout << "Thread " << i << "th running" << std::endl;

	while (!bQuit)
	{
		DWORD dwWaitResult = WaitForSingleObject(
			hControlPointMutex[i],    // handle to mutex
			1);  // no time-out interval

		switch (dwWaitResult)
		{
			// The thread got ownership of the mutex
		case WAIT_OBJECT_0:
			__try {
				auto start = std::chrono::high_resolution_clock::now();

				int nVal = (int)((double)rand() / (double)RAND_MAX * 255.0);
				memset(pImage, nVal, nPayload / 3);
				nVal = (int)((double)rand() / (double)RAND_MAX * 255.0);
				memset((BYTE*)pImage + nPayload / 3, nVal, nPayload / 3);
				nVal = (int)((double)rand() / (double)RAND_MAX * 255.0);
				memset((BYTE*)pImage + 2 * nPayload / 3, nVal, nPayload / 3);

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
				SetEvent(hControlPointEvent[i]);
			}
			break;

			// The thread got ownership of an abandoned mutex
			// The database is in an indeterminate state
		case WAIT_TIMEOUT:
			std::cout << "@@@ Thread " << i << "is waiting for mutex..." << std::endl;
			break;
			// The thread got ownership of an abandoned mutex
					// The database is in an indeterminate state
		case WAIT_ABANDONED:
			break;

		}

		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}

	CloseHandle(hMMFImage);
	UnmapViewOfFile(pImage);

	free(pData);
	std::cout << "Quitting!\n";
}

void Start()
{
	for (size_t i = 0; i < NUM_CONTROL_POINTS; i++)
	{
		threads[i] = std::thread(controlPointThreadFunc, i);
		//std::this_thread::sleep_for(std::chrono::milliseconds(30));
	}
}
void Configure()
{
	cpListMMF->state = ListState::UPDATE_PENDING;

	ZeroMemory(cpListMMF->points, sizeof(controlPoint) * 1024);
	cpListMMF->numPunti = NUM_CONTROL_POINTS;
	int nRand = rand();
	for (size_t i = 0; i < NUM_CONTROL_POINTS; i++)
	{
		cpListMMF->points[i].status = PointState::UPDATE_PENDING;

		cpListMMF->points[i].idPunto = nRand + i;
		cpListMMF->points[i].sizeX = 512;
		cpListMMF->points[i].sizeY = 512;
		cpListMMF->points[i].bpp = 24;
		wcscpy_s(cpListMMF->points[i].pathModello, 512, TEXT("D:\\modello.onnx"));
		swprintf_s(cpListMMF->points[i].mutexName, 128, TEXT("CP_%u_MUTEX"), cpListMMF->points[i].idPunto);
		swprintf_s(cpListMMF->points[i].eventReadyName, 128, TEXT("CP_%u_EVENTREADY"), cpListMMF->points[i].idPunto);

		if (hControlPointEvent[i] != NULL)
			CloseHandle(hControlPointEvent[i]);
		hControlPointEvent[i] = NULL;
		hControlPointEvent[i] = CreateEvent(NULL, FALSE, FALSE, cpListMMF->points[i].eventReadyName);


	}
	std::cout << "Configure OK. Set event!\n";
	bCanStart = true;
	SetEvent(hListEvent);
}

int main()
{
	std::cout << "DELTA VISIONE - Test AI via MMF + ONNX!\n";

	srand(1792);

	hListMutex = CreateMutex(NULL, FALSE, TEXT("LISTMUTEX"));
	hListEvent = CreateEvent(NULL, FALSE, FALSE, TEXT("LISTEVENTACK"));

	ZeroMemory(hControlPointMutex, sizeof(HANDLE) * NUM_CONTROL_POINTS);
	ZeroMemory(hControlPointEvent, sizeof(HANDLE) * NUM_CONTROL_POINTS);

	hcpListMMF = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, sizeof(controlPointsList), L"CONTROLPOINTLIST");
	DWORD dwErr = GetLastError();
	if (dwErr == ERROR_SUCCESS || dwErr == ERROR_ALREADY_EXISTS)
		cpListMMF = (PTcontrolPointsList)MapViewOfFile(hcpListMMF, FILE_MAP_WRITE, 0, 0, sizeof(controlPointsList));
	if (cpListMMF != NULL)
	{
		cpListMMF->state = ListState::CONFIGURED;
		wcscpy_s(cpListMMF->listMutexName, 128, TEXT("LISTMUTEX"));
		wcscpy_s(cpListMMF->listEventAckName, 128, L"LISTEVENTACK");
	}

	std::cout << "Press 'q' to quit, press 'c' to configure, press 's' to start sim" << std::endl;
	char c = _getch();
	while (!bQuit)
	{
		switch (c)
		{
		case 'c':
		{
			Configure();
		}
		break;
		case 's':
		{
			if (bCanStart)
				Start();
			else
				std::cout << "### Configure before start" << std::endl;
		}
		break;
		case 'q':
		{
			bQuit = true;
		}
		break;
		default:
			break;
		}
		if (!bQuit)
			c = _getch();
	}

	for (size_t i = 0; i < NUM_CONTROL_POINTS; i++)
	{
		threads[i].join();
		CloseHandle(hControlPointMutex[i]);
		CloseHandle(hControlPointEvent[i]);
	}

	CloseHandle(hListMutex);
	CloseHandle(hListEvent);
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
