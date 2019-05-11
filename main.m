%% VAD using FE and SVM

clc;clear;close all;

%% Input
[s,Fs]=audioread('wav\sa1.wav');

%% Adding Noise
disp('ADDING NOISE TO SPEECH');
snr = 5;
[ sn, fs] = audioread( 'noise_wav\train.wav' );
sn=sn(:,1);
noise = srconv(sn,fs,Fs); %% Sample rate conversion
  
% get mixture speech at a desired SNR level
noisy = addnoise( s, noise(1:length(s)), snr, fs );
L = noisy;
%% Labels for SVM
disp('CREATING LABELS FOR SVM');
[ S ] = read_wrd( 'wav\sa1.wrd' );
li  = extractLabel(length(L),S);
%% Dividing noisy speech into frames
[r, c]=size(L);
for i=1:r
   for j=1:c
      if(L(i,j)==0)
        L(i,j)=0.000001;
      end    
   end
end


x = L./max(abs(L));
T =length(x)/fs;
N = round(T*fs);                        % signal duration (samples)
n = [ 0:N-1 ];                          % discrete-time index vector
time = n/fs;                            % time vector
w = hanning(N).';                       % analysis window samples
nfft = 2^nextpow2(2*N);                 % FFT analysis length
freq = [ 0:nfft-1 ]/nfft*fs-fs/2;       % frequency vector (Hz)

f = [ 500 1500 3500 ];                  % vector of frequency components (Hz)
phi = [ pi/4 0 pi/3 ];                  % vector of phases (rad)
A = [ 0.125 1 0.5 ];                    % vector of amplitudes 


    alpha = 0.97;                       % preemphasis coefficient
    M = 20;                             % number of filterbank channels 
    C = 12;                             % number of cepstral coefficients
    Lc = 22;                            % cepstral sine lifter parameter
    LF = 300;                           % lower frequency limit (Hz)
    HF = 3700;                          % upper frequency limit (Hz)

    Tw = 32;                            % analysis frame duration (ms)
    Ts = Tw/4;                          % analysis frame shift (ms)
    Nw = round( 0.001*Tw*fs );          % analysis frame duration (samples)
    Ns = round( 0.001*Ts*fs );          % analysis frame shift (samples)
    direction = 'rows';                 % frames as rows
    padding = true;                     % pad with zeros
 
    gnl_hanning = @(L,S)(0.5*sqrt(2*S/(0.75*L))*(1-cos((2*pi*((0:L-1)+0.5))/L)));

    window = gnl_hanning(Nw,Ns);            % window function samples

    % divide signal to frames
    [ frames, indexes ] = vec2frames( x, Nw, Ns, direction, window, padding );
    [ Label, indexes_l ] = vec2frames( li, Nw, Ns, direction, window, padding );


    [m, n]=size(Label);
for i=1:m
    if Label(i,:)==0
        label(i,1)=0;
    else
        label(i,1)=1;
    end
end

% y=frames;
%% FEATURE EXTRACTION
disp('FEATURE EXTRACTION');
[rf, rc] = size(frames);
% Calculating Fuzzy Entropy
for i=1:rf
    SD(i) = std(frames(i,:));
    
    FUZZEN(i) = FuzzyEn(frames(i,:),2,0.2*SD(i),2);
  
end

%%% SVM - VAD 
%%% k-Fold cross validation evaluating performance


DATAs=FUZZEN';
groups=label;
k=10;

kk=0;
kk1=0;
[m1,n1,z1]=size(groups);
for i=1:m1
    if(groups(i)==1)
        kk=kk+1;
    end
    if(groups(i)==0)
     kk1=kk1+1;
    end

end

kks=(kk/10)*6;

DATA = zeros(size(FUZZEN'));
for i=m1:m1+(kks+kk1+kk)
    groups(i)=0;
    groups(i+1)=0;
    DATA(i)=0;
    DATA(i+1)=0;
    DATAs(i,:)=DATA(1,:);
    DATAs(i+1,:)=DATA(2,:);
    i=i+1;
end  

kk=0;
kk1=0;
[m1,n1,z1]=size(groups);
for i=1:m1
    if(groups(i)==1)
        kk=kk+1;
    end
    if(groups(i)==0)
    kk1=kk1+1;
    end
end

sss=[DATAs groups];
[cR,cC]=size(sss);
sss(:,1)=sss(:,1)/max(sss(:,1));
[sss1,indi]=sort(sss(:,2));
sss2 = sss(indi,:);
groups1=sss2(:,cC); % Sorted label
DATA_SVM=sss2(:,1:cC-1); % Sorted data
[mx,nx,zx]=size(DATA_SVM);
ggg(1,1)=0;

%% SVM-Classification

disp('SVM-CLASSIFIER');
cvFolds = crossvalind('Kfold', groups1, k);   %# get indices of 10-fold CV
cp = classperf(groups1);                      %# init performance tracker

for i = 1:k                                  %# for each fold
    testIdx = (cvFolds == i);                %# get indices of test instances
    trainIdx = ~testIdx;                     %# get indices training instances

    %# train an SVM model over training instances
    options = optimset('maxiter',1000);
    svmModel = svmtrain(DATA_SVM(trainIdx,:), groups1(trainIdx), ...
                 'Autoscale',true, 'Showplot',false, 'Method','QP','quadprog_opts' , options,...
                 'BoxConstraint',2e-1, 'Kernel_Function','rbf', 'RBF_Sigma',1);

    %# test using test instances
    pred = svmclassify(svmModel, DATA_SVM(testIdx,:), 'Showplot',false);
       
    %# evaluate and update performance object
    cp = classperf(cp, pred, testIdx);
    
end

%% RESULT ANALYSIS

ACC = cp.CorrectRate;
ERR = cp.ErrorRate;
CM = cp.CountingMatrix;
SENSI = cp.Sensitivity;
SPECI = cp.Specificity;



