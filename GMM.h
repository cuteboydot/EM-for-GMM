#pragma once
#include "stdafx.h"
#include "targetver.h"
#include <math.h>

typedef struct inputdata_multi_gauss {
	double * pData;
	int nClass;
	double * pNormalProb;
} INPUTDATA_MULTI_GAUSS;

class CGMM
{
public:
	CGMM(void);
	~CGMM(void);

	void init(int nSizeK, int nSizeRecord, int nSizeFeature, INPUTDATA_MULTI_GAUSS ** ppDataList);
	void train();

private:
	void i_step();
	void e_step();
	void m_step();
	inline double getgauss(double dMean, double dVar, double dValue);
	void printparameter();

	int m_nMaxLoop;

	// datas for training
	int m_nSizeK;							// number of output pattern
	int m_nSizeFeature;						// number of feature(column)
	int m_nSizeRecord;						// number of records
	INPUTDATA_MULTI_GAUSS ** m_ppDataList;	// input datas of records

	bool m_bIsChangeClass;

	int * m_pNumClass;						// number of records in each class
	double ** m_ppSumFeatClass;				// sum value of feature value in each class
	double ** m_ppSumVarClass;				// sum value of feature variation square in each class
	double ** m_ppMeanFeatClass;			// mean value of feature in each class
	double ** m_ppVarFeatClass;				// covariation value of feature in each class

	double * m_pProbClass;					// 𝑷(𝒄)
};

