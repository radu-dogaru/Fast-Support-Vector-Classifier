% XFSVC.M 
% Accelerated version of FSVC (using .MEX) 
% Credit for .MEX implementation goes to Ioana DOGARU and Radu DOGARU (Polytehnic University of Bucharest) 
% Tested under Octave 4.0  (default .MEX files) and under Matlab 7.7.0 (32 bit) .mex-es included in a directory)
% Tested on Microsoft Windows XP, Windows 7, operating systems 
% Algorithm description in relevant publications (see the list bellow) 
% TRAINING SPEEDS COMPARABLE AND SOMETIMES LESS THAN SVM / ELM implementas on the same computing platform 
% Included datasets are processed versions of datasets IRIS, SATIMG (Statlog) and PHONEME 
% from ELENA project (https://www.elen.ucl.ac.be/neural-nets/Research/Projects/ELENA/databases/REAL/)
% and OPTD64 from https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits
 
%--------------------------------------------------------------------------
% Code implementing Fast Support Vector Classifier (FSVC) a.k.a. 
% RBF-M (Modified Radial Basis Functions Network) 
% 
% Speedups may be as high as 2 orders of magnitude when compared to 
% pure Matlab implementation (particularly for large number of hidden units) 
%--------------------------------------------------------------------------
%LICENSE 
%Copyright (c) 2016, Radu Dogaru and Ioana Dogaru
%High Performance and Natural Computing Lab.  
%http://atm.neuro.pub.ro/radu_d/  
%radu_d@ieee.org 

%All rights reserved.

%Redistribution and use in source and binary forms, with or without
%modification, are permitted provided that the following conditions are
%met:

%    * Redistributions of source code must retain the above copyright
%      notice, this list of conditions and the following disclaimer.
%    * Redistributions in binary form must reproduce the above copyright
%      notice, this list of conditions and the following disclaimer in
%      the documentation and/or other materials provided with the distribution

%THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
%AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
%IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
%ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
%LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
%CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
%SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
%INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
%CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
%ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
%POSSIBILITY OF SUCH DAMAGE.

%---------------------------------------------------------------------------------------
%Please cite relevant publications in any published work that benefits from this module:

%[1]
%Dogaru, R. ; Murgan, A.T. ; Ortmann, S. ; Glesner, M.  
%"A modified RBF neural network for efficient current-mode VLSI implementation"
%Microelectronics for Neural Networks, 1996., Proceedings of Fifth International Conference on
%Digital Object Identifier: 10.1109/MNNFS.1996.493801
%Publication Year: 1996 , Page(s): 265 - 270

%[2] 
%Dogaru, R. ; Dogaru, I.
%"An efficient finite precision RBF-M neural network architecture using support vectors"
%Neural Network Applications in Electrical Engineering (NEUREL), 2010 10th Symposium on
%Digital Object Identifier: 10.1109/NEUREL.2010.5644089
%Publication Year: 2010 , Page(s): 127 - 130

%[3]
%R. Dogaru, "A hardware oriented classifier with simple constructive
%training based on support vectors", in Proceedings of CSCS-16, the 
%16th Int'l Conference on Control Systems and Computer Science, 
%May 22 - 26, 2007, Bucharest, Vol.1, pp. 415-418.
 
% [4] 
% R. Dogaru and Ioana Dogaru, "A Super Fast Vector Support Classifier Using 
% Novelty Detection and no Synaptic Tuning", 
% in Proceedings COMM-2016 (in press). 
%--------------------------------------------------------------------------
% % Last update, May 24, 2016  
% -------------------------------------------------------------------------

%================================================================
function [conf_best, accur_best, Wbest, cen, ep_best, ncen]=xfsvc(prag,raza,eta,tip_r,dist_type,problem,epoci)
%==========================================================================
%-- OUTPUTS -------
% conf_best (best confusion matrix) 

% acsur_best (best accuracy) 

% Wbest - Adaline parameters (for best generalization solution) 

% cen - centers (support vectors) for solution   

% ep_best - epoch when best generalization was obtained. 

% ncen - number of support vectors
% - INPUTS ---- 

% prag = neuron activation threshold (usually 1)  

% raza = kernel radius (must be tuned for best performance) 

% eta = learning rate (e.g. 1/16 etc.).  

% tip_r = 'rbf_dog' (traingular kernel) 'rbf_gus' (Gaussian kernel)

% dist_type = 'manh' - norm1 - Manhattan 'eucl' Euclidean 

% problema = problem file name (e.g. 'iris1') previously saved in a LIBSVM 
% format: Samples Labels matrices per each _train and _test files. 

% epoci = maximal number of epochs 

% NOTE: Training data should be randomized ! (no consecutive samples in the same class) 
% --- USAGE EXAMPLES --------------------------------------------------------------------------------
% multi-class 
% [conf_best, accur_best, Wbest, cen, ep_best,ncen]=xfsvc(2,1.8,0.1,'rbf_dog','manh','satimg',4);
% [conf_best, accur_best, Wbest, cen, ep_best,ncen]=xfsvc(1,1.3,0.1,'rbf_dog','manh','iris1',20);

% multi-class (no tuning, see [4]) 
%[conf_best, accur_best, Wbest, cen, ep_best,ncen]=xfsvc(4,1.35,0/7,'rbf_dog','manh','satimg',15);
% Number of RBF units =1330
%-------------------------------------
%Total training time: 0.57812 seconds
%=======================================================================================
%FSVC predict (MEX) implementation - Copyright Radu DOGARU and Ioana DOGARU NHC Lab.
%http://atm.neuro.pub.ro/radu_d/Brief_pres_NHC_lab.htm#Efficient_classifier_for_embedded
%=======================================================================================
%Total testing time: 0.40625 seconds
%Accuraccy=89.372%

% single class 
% [conf_best, accur_best, Wbest, cen, ep_best,ncen]=xfsvc(1,.27,1/16,'rbf_dog','manh','phoneme',15);
% [conf_best, accur_best, Wbest, cen, ep_best,ncen]=xfsvc(1,.12,0/16,'rbf_dog','manh','phoneme',10); conf_best,


eval(['load ',problem,'_train']); 
Samples1=Samples; 
Labels1=Labels;

[n Ntr]=size(Samples1); 

tic; 
HS1=Samples1; 

if tip_r=='rbf_gus'
    tiprbf=2; 
elseif  tip_r=='rbf_dog'
    tiprbf=1; 
end

    
if dist_type=='eucl'
    tipdis=2; 
elseif dist_type =='manh'
    tipdis=1; 
end


[inW outW Tix] = mexFSVCxTrain(HS1, Labels1', raza, 1, prag, tiprbf, tipdis, eta, epoci); 
% Tix is a list of indexes to locate the support vectors 


disp(['tip RBF: (1/triangular 2/Gaussian): ',num2str(tiprbf)]); 
disp(['tip distance (1/Manhattan (L1) 2/Euclidean (L2)) : ',num2str(tipdis)]); 
disp(['threshold=',num2str(prag)]);
disp(['RBF kernel radius =',num2str(raza)]);
disp(['LMS training rate =',num2str(eta)]);
disp(['Epochs of LMS training =',num2str(epoci)]);

disp(['Number of RBF units =',num2str(size(inW,1))]);
disp('-------------------------------------');

disp(['Total training time: ',num2str(toc), ' seconds']); 

HStr=HS1; 
eval(['load ',problem,'_test']); 

tic;
HS1=Samples; 

scores = mexFSVCxPredict(inW, raza, outW, HS1, tiprbf, tipdis );

disp(['Total testing time: ',num2str(toc), ' seconds']); 



[nout, Ntest]=size(scores);
% evalueaza acuratetea 
conf=zeros(nout,nout); 
for tt=1:Ntest
    Ytt=wintakeall(scores(:,tt));  
    ix=find(Ytt==1); 
    i_pred=ix(1); 
    i_actual=Labels(tt); 
    conf(i_actual,i_pred)=1+conf(i_actual,i_pred); 
end

%adaugat 03/06/2015
    if nout==2
    conf_best=conf;
    TP=conf_best(1,1); TN=conf_best(2,2); FN=conf_best(2,1); FP=conf_best(1,2); 
    PREC=TP/(TP+FP); disp(['Precision is: ',num2str(100*PREC),'%']); 
    RECALL=TP/(TP+FN); disp(['Recall is: ',num2str(100*RECALL),'%']); 
    ACCURACY=(TP+TN)/(TP+TN+FP+FN); disp(['Accuracy is: ',num2str(100*ACCURACY),'%']); 
    F2=5*PREC*RECALL/(4*PREC+RECALL); disp(['F2-score is: ',num2str(100*F2),'%']); 
    else 
    ACCURACY=sum(diag(conf))/sum(sum(conf));    
    end
 
conf_best=conf; 
accur_best= ACCURACY; 
Wbest = outW; 
cen = inW; 
ep_best=epoci; 
ncen = num2str(size(inW,1)); 

disp(['Accuraccy=',num2str(100*ACCURACY),'%'])
disp('-------------------------------------');

    
 