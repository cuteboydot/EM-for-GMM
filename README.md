# EM-for-GMM
Implementation of EM using K-Means(Gaussian Mixture Model)  
cuteboydot@gmail.com  

reference : https://www.cs.duke.edu/courses/fall07/cps271/EM.pdf  

E-step :  
Estimate distribution over labels given a certain fixed model.  
M-step :  
Choose new parameters for model to maximize expected log-likelihood of observed data and hidden variables.  

- example : male or female  
<br>
<img src="https://github.com/cuteboydot/Em-for-GMM/blob/master/img/maleorfemale.JPG" />
</br>

- example : test result  
<br>
<img src="https://github.com/cuteboydot/Em-for-GMM/blob/master/img/em_result.JPG" />
</br>

- usage : train  
```cpp  
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
```
- usage details: E-step  
```cpp  
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
}
```
- usage details: M-step  
```cpp  
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
```
