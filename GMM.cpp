#include "GMM.h"


CGMM::CGMM(void)
{
	m_nMaxLoop = 10000;

	m_pNumClass = 0;
	m_pProbClass = 0;

	m_ppSumFeatClass = 0;
	m_ppSumVarClass = 0;
	m_ppMeanFeatClass = 0;
	m_ppVarFeatClass = 0;
}

CGMM::~CGMM(void)
{
	delete[] m_pNumClass;
	delete[] m_pProbClass;

	for(int a=0; a<m_nSizeFeature; a++) {
		delete[] m_ppSumFeatClass[a];
		delete[] m_ppSumVarClass[a];
		delete[] m_ppMeanFeatClass[a];
		delete[] m_ppVarFeatClass[a];
	}
	delete[] m_ppSumFeatClass;
	delete[] m_ppSumVarClass;
	delete[] m_ppMeanFeatClass;
	delete[] m_ppVarFeatClass;
}

void CGMM::init(int nSizeK, int nSizeRecord, int nSizeFeature, INPUTDATA_MULTI_GAUSS ** ppDataList)
{
	m_nSizeK = nSizeK;
	m_nSizeRecord = nSizeRecord;
	m_nSizeFeature = nSizeFeature;
	m_ppDataList = ppDataList;

	m_pNumClass = new int[m_nSizeK];
	m_pProbClass = new double[m_nSizeK];
	m_ppSumFeatClass = new double*[m_nSizeK];
	m_ppSumVarClass = new double*[m_nSizeK];
	m_ppMeanFeatClass = new double*[m_nSizeK];
	m_ppVarFeatClass = new double*[m_nSizeK];
	for(int a=0; a<m_nSizeK; a++) {
		m_pNumClass[a] = 0;
		m_pProbClass[a] = 0;

		m_ppSumFeatClass[a] = new double[m_nSizeFeature];
		m_ppSumVarClass[a] = new double[m_nSizeFeature];
		m_ppMeanFeatClass[a] = new double[m_nSizeFeature];
		m_ppVarFeatClass[a] = new double[m_nSizeFeature];
		for(int b=0; b<m_nSizeFeature; b++) {
			m_ppSumFeatClass[a][b] = 0;
			m_ppSumVarClass[a][b] = 0;
			m_ppMeanFeatClass[a][b] = 0;
			m_ppVarFeatClass[a][b] = 0;
		}
	}
}

void CGMM::train()
{
	i_step();

	for(int z=0; z<m_nMaxLoop; z++) {
		m_step();
		if(!m_bIsChangeClass)
			break;
		e_step();
	}
	printparameter();
}

// initalize centroid
void CGMM::i_step()
{
	for(int a=0; a<m_nSizeRecord; a++) {
		// assign initial cluster(random)
		m_ppDataList[a]->nClass = a % m_nSizeK;
	}

	e_step();
}

// update probability parameters
void CGMM::e_step()
{
	// reset statistics
	for(int a=0; a<m_nSizeK; a++) {
		m_pNumClass[a] = 0;
		for(int b=0; b<m_nSizeFeature; b++) {
			m_ppSumFeatClass[a][b] = 0; 
			m_ppSumVarClass[a][b] = 0;
		}
	}

	// count record & feature value
	for(int a=0; a<m_nSizeRecord; a++) {
		m_pNumClass[m_ppDataList[a]->nClass]++;
		for(int b=0; b<m_nSizeFeature; b++) {
			m_ppSumFeatClass[m_ppDataList[a]->nClass][b] += m_ppDataList[a]->pData[b]; 
		}
	}

	// calc mean
	for(int a=0; a<m_nSizeK; a++) {
		m_pProbClass[a] = (double)((double)m_pNumClass[a] / (double)m_nSizeRecord);
		for(int b=0; b<m_nSizeFeature; b++) {
			m_ppMeanFeatClass[a][b] = (double)((double)m_ppSumFeatClass[a][b] / (double)m_pNumClass[a]);
		}
	}

	// calc variance
	for(int a=0; a<m_nSizeRecord; a++) {
		for(int b=0; b<m_nSizeFeature; b++) {
			m_ppSumVarClass[m_ppDataList[a]->nClass][b] = m_ppSumVarClass[m_ppDataList[a]->nClass][b] + 
				(m_ppDataList[a]->pData[b] - m_ppMeanFeatClass[m_ppDataList[a]->nClass][b]) * 
				(m_ppDataList[a]->pData[b] - m_ppMeanFeatClass[m_ppDataList[a]->nClass][b]);
		} 
	}

	for(int a=0; a<m_nSizeK; a++) {
		for(int b=0; b<m_nSizeFeature; b++) {
			m_ppVarFeatClass[a][b] = m_ppSumVarClass[a][b] / (double)(m_pNumClass[a] - 1);
		}
	}	

	// print statistics
	for(int a=0; a<m_nSizeK; a++) {
		printf("cluster[%d] cnt:%d, ", a, m_pNumClass[a]); 
		
		printf("data[");
		for(int z=0; z<m_nSizeRecord; z++) {
			if(m_ppDataList[z]->nClass == a)
				printf("%d ", z);
		}
		printf("]\n");

		printf("mean:  ");
		for(int b=0; b<m_nSizeFeature; b++) {
			printf("feat[%d]:%.1f/%d, ", b, m_ppSumFeatClass[a][b], m_pNumClass[a]); 
		}
		printf("\n");
		printf("var:   ");
		for(int b=0; b<m_nSizeFeature; b++) {
			printf("feat[%d]:%.1f/(%d-1), ", b, m_ppSumVarClass[a][b], m_pNumClass[a]); 
		}
		printf("\n");
	}
	printf("\n");
}

// assign clustering
void CGMM::m_step()
{
	double * pProbability = new double[m_nSizeK];
	double dGauss = 1, dProbSum = 0;
	
	m_bIsChangeClass = false;

	for(int a=0; a<m_nSizeRecord; a++) {
		dProbSum = 0;

		for(int b=0; b<m_nSizeK; b++) {
			pProbability[b] = 1;
			dGauss = 1;
			
			for(int c=0; c<m_nSizeFeature; c++) {
				dGauss = getgauss(m_ppMeanFeatClass[b][c], m_ppVarFeatClass[b][c], m_ppDataList[a]->pData[c]);
				pProbability[b] *= dGauss;
			}
			dProbSum += pProbability[b];
		}

		double dTemp = 0;
		int nMaxId = 0;
		for(int b=0; b<m_nSizeK; b++) {
			m_ppDataList[a]->pNormalProb[b] = pProbability[b] / dProbSum;

			int nCurClass = m_ppDataList[a]->nClass;
			if(dTemp < pProbability[b]) {
				nMaxId = b;
				dTemp = pProbability[b];
			}
		}

		if(m_ppDataList[a]->nClass != nMaxId)
			m_bIsChangeClass = true;
		m_ppDataList[a]->nClass = nMaxId;

	}

	delete[] pProbability;
}

inline double CGMM::getgauss(double dMean, double dVar, double dValue)
{
	double dGauss = 1;
	const double dPi = 3.14159265358979323846;

	dGauss = (1 / sqrt(2 * dPi * dVar)) * (exp((-1 * (dValue - dMean) * (dValue - dMean)) / (2*dVar)));

	return dGauss;
}

void CGMM::printparameter()
{
	for(int a=0; a<m_nSizeK; a++) {
		printf("P(c%d) = %0.3f \n", a, m_pProbClass[a]);
		for(int b=0; b<m_nSizeFeature; b++) {
			printf("Mean[%d][%d]=%.4f,\tVariance[%d][%d]=%.4f \n", a, b, m_ppMeanFeatClass[a][b], a, b, m_ppVarFeatClass[a][b]);
		}
	}
	printf("\n");
}