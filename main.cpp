#include <iostream>
#include <fstream>
#include <conio.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <stdio.h>
#include <float.h>

#include "GMM.h"

/**
EXAMPLE
|---------------------------------------|
|Num	|Height	|Weight	|foot	|Class	|
|-------|-------|-------|-------|-------|
|1		|6		|180	|12		|?   	|
|2		|5.92	|190	|11		|?   	|
|3		|5.58	|170	|12		|?   	|
|4		|5.92	|165	|10		|?   	|
|5		|5		|100	|6		|?   	|
|6		|5.5	|150	|8		|?   	|
|7		|5.42	|130	|7		|?   	|
|8		|5.75	|150	|9		|?   	|
|9		|6		|130	|8		|?   	|
|---------------------------------------|
**/

void main()
{
#define SIZE_RECORD		9
#define SIZE_OUTPUT		2
#define SIZE_FEATURE	3

	enum ANSWERLIST	{MALE=0, FEMALE};

	INPUTDATA_MULTI_GAUSS ** ppInputData;

	ppInputData = new INPUTDATA_MULTI_GAUSS*[SIZE_RECORD];

	ppInputData[0] = new INPUTDATA_MULTI_GAUSS;
	ppInputData[0]->pNormalProb = new double[SIZE_OUTPUT];
	ppInputData[0]->pData = new double[SIZE_FEATURE];
	ppInputData[0]->pData[0] = 6;
	ppInputData[0]->pData[1] = 180;
	ppInputData[0]->pData[2] = 12;
	ppInputData[0]->nClass = -1;

	ppInputData[1] = new INPUTDATA_MULTI_GAUSS;
	ppInputData[1]->pNormalProb = new double[SIZE_OUTPUT];
	ppInputData[1]->pData = new double[SIZE_FEATURE];
	ppInputData[1]->pData[0] = 5.92;
	ppInputData[1]->pData[1] = 190;
	ppInputData[1]->pData[2] = 11;
	ppInputData[1]->nClass = -1;

	ppInputData[2] = new INPUTDATA_MULTI_GAUSS;
	ppInputData[2]->pNormalProb = new double[SIZE_OUTPUT];
	ppInputData[2]->pData = new double[SIZE_FEATURE];
	ppInputData[2]->pData[0] = 5.58;
	ppInputData[2]->pData[1] = 170;
	ppInputData[2]->pData[2] = 12;
	ppInputData[2]->nClass = -1;

	ppInputData[3] = new INPUTDATA_MULTI_GAUSS;
	ppInputData[3]->pNormalProb = new double[SIZE_OUTPUT];
	ppInputData[3]->pData = new double[SIZE_FEATURE];
	ppInputData[3]->pData[0] = 5.92;
	ppInputData[3]->pData[1] = 165;
	ppInputData[3]->pData[2] = 10;
	ppInputData[3]->nClass = -1;

	ppInputData[4] = new INPUTDATA_MULTI_GAUSS;
	ppInputData[4]->pNormalProb = new double[SIZE_OUTPUT];
	ppInputData[4]->pData = new double[SIZE_FEATURE];
	ppInputData[4]->pData[0] = 5;
	ppInputData[4]->pData[1] = 100;
	ppInputData[4]->pData[2] = 6;
	ppInputData[4]->nClass = -1;

	ppInputData[5] = new INPUTDATA_MULTI_GAUSS;
	ppInputData[5]->pNormalProb = new double[SIZE_OUTPUT];
	ppInputData[5]->pData = new double[SIZE_FEATURE];
	ppInputData[5]->pData[0] = 5.5;
	ppInputData[5]->pData[1] = 150;
	ppInputData[5]->pData[2] = 8;
	ppInputData[5]->nClass = -1;

	ppInputData[6] = new INPUTDATA_MULTI_GAUSS;
	ppInputData[6]->pNormalProb = new double[SIZE_OUTPUT];
	ppInputData[6]->pData = new double[SIZE_FEATURE];
	ppInputData[6]->pData[0] = 5.42;
	ppInputData[6]->pData[1] = 130;
	ppInputData[6]->pData[2] = 7;
	ppInputData[6]->nClass = -1;

	ppInputData[7] = new INPUTDATA_MULTI_GAUSS;
	ppInputData[7]->pNormalProb = new double[SIZE_OUTPUT];
	ppInputData[7]->pData = new double[SIZE_FEATURE];
	ppInputData[7]->pData[0] = 5.75;
	ppInputData[7]->pData[1] = 150;
	ppInputData[7]->pData[2] = 9;
	ppInputData[7]->nClass = -1;

	ppInputData[8] = new INPUTDATA_MULTI_GAUSS;
	ppInputData[8]->pNormalProb = new double[SIZE_OUTPUT];
	ppInputData[8]->pData = new double[SIZE_FEATURE];
	ppInputData[8]->pData[0] = 6;
	ppInputData[8]->pData[1] = 130;
	ppInputData[8]->pData[2] = 8;
	ppInputData[8]->nClass = -1;

	// train
	CGMM * pGmm = new CGMM();
	pGmm->init(SIZE_OUTPUT, SIZE_RECORD, SIZE_FEATURE, ppInputData);
	pGmm->train();

	// print results
	for(int a=0; a<SIZE_RECORD; a++) {
		printf("data#%d => class[%d] :", a, ppInputData[a]->nClass);
		for(int b=0; b<SIZE_OUTPUT; b++) {
			printf(" [%d]%.3f ", b, ppInputData[a]->pNormalProb[b]);
		}
		printf("\n");
	}
	printf("\n");
	printf("-----------------------------------------------------\n\n");

	// terminate memory	
	for(int a=0; a<SIZE_RECORD; a++) {
		if(ppInputData[a]) {
			delete[] ppInputData[a]->pData;
			delete[] ppInputData[a]->pNormalProb;
			delete[] ppInputData[a];
		}
	}
	delete[] ppInputData;

	printf("Bye~~~!!! \n");
}