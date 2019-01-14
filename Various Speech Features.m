%%%%%%%%  EEE-598 Speech and Audio Processing & Perception %%%%%%%%%%%%%%
%%%%%%%%%%%%%% Speech Signals Features extraction %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;
clear all;
close all;

SNR_all = zeros(5,5);

Audio_files = {'TIMIT/sx108.wav','TIMIT/si1818.wav','TIMIT/si1188.wav','TIMIT/si558.wav','TIMIT/sa2.wav'}; % TIMIT Audio Files
Speech_files = {'Q7speech/Track 49.wav','Q7speech/Track 50.wav','Q7speech/Track 51.wav','Q7speech/Track 52.wav','Q7speech/Track 53.wav', 'Q7speech/Track 54.wav'}; % Q7Speech Audio files
Music_files = {'Q7music/Track 55.wav','Q7music/Track 56.wav','Q7music/Track 57.wav','Q7music/Track 58.wav','Q7music/Track 59.wav', 'Q7music/Track 60.wav'}; % Q7Music Audio files
    
for i=1:5    
    [y{i}, Fs{i}] = audioread(Audio_files{i},'native'); % Reading the files
    r1 = changesamp(double(y{i})); % Function to change the sampling rate 
    r2 = changebps(double(y{i})); % Functio to change the bit-rate per sample
    SNR_all(i,:)= snr2(double(y{i})); % Function to calculate the SNR between the original and reduced bit_rate signal
    lowfilter = lpf(double(y{i}), Fs{i}); % Function to calulate High Frequency Hearing loss 
end

combo = {};
combo2 ={};
combo3 ={};
combo4 ={};
combo5 ={};
combom = {};
combo2m ={};
combo3m ={};
combo4m ={};
combo5m ={};

for i=1:6
    [y{i}, Fs{i}] = audioread(Speech_files{i}); % Reading the Speech files
    [ym{i}, Fsm{i}] = audioread(Music_files{i}); % Reading the Music files
    
    low_energy = lefc(y{i}, Fs{i}); % Percentage of low-energy frames of RMS
    low_energym = lefc(ym{i}, Fsm{i}); 
    
    spec_rolloff = sr(y{i}, Fs{i}); % Spectral Roll-off,ie, 95th Percentile of PSD
    spec_rolloffm = sr(ym{i}, Fsm{i}); 
    
    spec_centroid = sc(y{i}, Fs{i}); % Spectral Centroid
    spec_centroidm = sc(ym{i}, Fsm{i});

    spec_flux = sf(y{i}, Fs{i}); % Spectral Flux
    spec_fluxm = sf(ym{i}, Fsm{i});
    
    zero_cross = zc(y{i}, Fs{i}); % Zero Crossings in each frame
    zero_crossm = zc(ym{i}, Fsm{i});
    
    combo = [combo low_energy];
    combom = [combom low_energym];
    
    combo2 = [combo2 spec_rolloff];
    combo2m = [combo2m spec_rolloffm];
    
    combo3 = [combo3 spec_centroid];
    combo3m = [combo3m spec_centroidm];
    
    combo4 = [combo4 spec_flux];
    combo4m = [combo4m spec_fluxm];
    
    combo5 = [combo5 zero_cross];
    combo5m = [combo5m zero_crossm];
    
end

%%%%%%%%%%% SPEECH SIGNALS FEATURES EXTRACTION %%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% Concatenating the Speech Features of all frames %%%%%%%
low_energy_frames = [combo{1}(1,1:end-6) combo{2}(1,1:end-6) combo{3}(1,1:end-6) combo{4}(1,1:end-6) combo{5}(1,1:end-6) combo{6}(1,1:end-6)]';
spec_rolloff = [combo2{1}(1,1:end-6) combo2{2}(1,1:end-6) combo2{3}(1,1:end-6) combo2{4}(1,1:end-6) combo2{5}(1,1:end-6) combo2{6}(1,1:end-6)]';
spec_centroid = [combo3{1}(1,1:end-6) combo3{2}(1,1:end-6) combo3{3}(1,1:end-6) combo3{4}(1,1:end-6) combo3{5}(1,1:end-6) combo3{6}(1,1:end-6)]';
spec_flux = [combo4{1}(1,1:end-6) combo4{2}(1,1:end-6) combo4{3}(1,1:end-6) combo4{4}(1,1:end-6) combo4{5}(1,1:end-6) combo4{6}(1,1:end-6)]';
zero_cross = [combo5{1}(1,1:end-6) combo5{2}(1,1:end-6) combo5{3}(1,1:end-6) combo5{4}(1,1:end-6) combo5{5}(1,1:end-6) combo5{6}(1,1:end-6)]';

%%%%%%%%%%% SPEECH SIGNALS FEATURES EXTRACTION %%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% Concatenating the Music Features of all frames %%%%%%%
low_energy_framesm = [combom{1}(1,1:end-6) combom{2}(1,1:end-6) combom{3}(1,1:end-6) combom{4}(1,1:end-6) combom{5}(1,1:end-6) combom{6}(1,1:end-6)]';
spec_rolloffm = [combo2m{1}(1,1:end-6) combo2m{2}(1,1:end-6) combo2m{3}(1,1:end-6) combo2m{4}(1,1:end-6) combo2m{5}(1,1:end-6) combo2m{6}(1,1:end-6)]';
spec_centroidm = [combo3m{1}(1,1:end-6) combo3m{2}(1,1:end-6) combo3m{3}(1,1:end-6) combo3m{4}(1,1:end-6) combo3m{5}(1,1:end-6) combo3m{6}(1,1:end-6)]';
spec_fluxm = [combo4m{1}(1,1:end-6) combo4m{2}(1,1:end-6) combo4m{3}(1,1:end-6) combo4m{4}(1,1:end-6) combo4m{5}(1,1:end-6) combo4m{6}(1,1:end-6)]';
zero_crossm = [combo5m{1}(1,1:end-7) combo5m{2}(1,1:end-7) combo5m{3}(1,1:end-7) combo5m{4}(1,1:end-7) combo5m{5}(1,1:end-7) combo5m{6}(1,1:end-7)]';

%%%%%%% Putting all 5 Features Together for Speech Signals %%%%%%%%%

Speech_AllFeatures = [low_energy_frames spec_rolloff spec_centroid spec_flux zero_cross];

%%%%%%% Putting all 5 Features Together for Music Signals %%%%%%%%%

Music_AllFeatures = horzcat(low_energy_framesm(1:end-6) ,spec_rolloffm(1:end-6), spec_centroidm(1:end-6), spec_fluxm(1:end-6) ,zero_crossm);

%%%%%%%% SIMILARITY BETWEEN THE SPEECH & MUSIC FEATURES %%%%%%%%%%%%%%%%%%

Speech_subset = Speech_AllFeatures(1:222,1:5);
Music_subset = Music_AllFeatures(1:222,1:5);
Similarity_Score = zeros(1,5);
for k=1:5
    Similarity_Score(k) = BCdistance(Speech_AllFeatures(:,k) ,Music_subset(:,k));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% 1. & 2. Reduce Sampling Rate of TIMIT Samples %%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function csamp = changesamp(X)
count=1;
    for fs=[12000 8000 4000 2000 1000]
        subplot(6,1,count);
        if fs == 12000
            r = resample(X,3,4);
        else
            r = resample(X,1,16000/fs); % Resampling Factor
        end

        csamp=double(r);
        z = fft(csamp); %% FFT of the signal
        z = fftshift(z); %% FFT Shift
        N11 = length(r);
        f = [-N11/2:N11/2-1]/N11;
        plot(f,abs(z)) % The Plot
        xlabel('Frequency (in hertz)');
        title(strcat('Magnitude Response of Signal'));
        count=count+1;
       
    end
        %figure;
        
%%%%%%%%%%%%%%%% Plotting % Correct words vs Sampling-Rate %%%%%%%%%%%%%%%
    x3 = [1,2,4,8,12,16];
    y3 = [0,0,0.1,0.8,1,1];
    y31 = [0,0,0.08,0.75,1,1];
    y32 = [0,0,0.2,0.8,1,1];
    y33 = [0,0,0.14,0.57,1,1];
    y34 = [0,0,0.25,0.625,1,1];

    plot(x3,y3,x3,y31, x3,y32,x3,y33,x3,y34);
    title('% Correct words vs Sampling-Rate');
    xlabel('Sampling Rate (in KHz)');
    ylabel('% Correct Words');
    legend({'sa2.wav','si558.wav','si1188.wav','si1818.wav','sx108.wav'})
    figure;

%% Conclusion -> As we can see, the input 16-bit encoded signal was resampled to 12,8,4,2 and 1KHz,
%% and after each down-sampling, the words drop, and by the time the sampling rate reached 2kHz, almost all
%% words could not be heard. This is because, as the sampling frequency decreases, the number of samples taken
%% per second decreases, and thus more and more of our signal gets lost. 

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
%%%%%%% 3. & 4. Reduce Bits per sample of TIMIT Samples %%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
function Change = changebps(X)
count=1;
    for fs=[12000 8000 4000 2000 1000]
        subplot(6,1,count);
        bits= 16 - fs/1000;
        X =X./rms(X);
        fx= sign(X).*((abs(X)-(mod(abs(X),2^bits))))/2;
        Change=double(fx);
        z = fft(Change); %% FFT of the signal
        z = fftshift(z); %% FFT Shift
        N11 = length(Change);
        f = [-N11/2:N11/2-1]/N11;
        plot(f,abs(z)) % The Plot
        xlabel(' Frequency (in Herts)');
        title(strcat('Magnitude Response of Signal'));
        count=count+1;
    end
        %figure;
    x3 = [1,2,4,8,12,16];
    y3 = [0,0,0,0,1,1];
    y31 = [0,0,0,0,1,1];
    y32 = [0,0,0,0,1,1];
    y33 = [0,0,0,0,1,1];
    y34 = [0,0,0,0,1,1];
    figure;
    plot(x3,y3,x3,y31, x3,y32,x3,y33,x3,y34);
    title('% Correct words vs Bits per sample (Bps)');
    xlabel('Bits per sample (Bps)');
    ylabel('% Correct Words');
    legend({'sa2.wav','si558.wav','si1188.wav','si1818.wav','sx108.wav'})

%% Conclusion -> As we can see, the input 16-bit encoded signal was compressed to 12,8,4,2 and 1 bit-per-sample,
%% without changing the quality of the signal, ie, by just truncating the amplitudes to the respective ranges.
%% For 16 bps, the signal can be heard very well, with 12bps, the whole sentence is heard but with lower resolution,
%% and with 8bps, some sound can be heard, though very faint, but with 4bps,2bps and 1bps, nothing could be heard.
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 5. SNR Calculation between each Quantized and original signal %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function SNR = snr2(X)
count=1;
    for fs=[12000 8000 4000 2000 1000]
        energy = sum((X).^2);
        bits= 16 - fs/1000;
       
%# normalize to -1...1
        fx= sign(X).*((abs(X)-(mod(abs(X),2^bits))))/2;
        fx_double=double(fx);

        energy_new_bps = mean((fx_double).^2);
        diff=energy-energy_new_bps;
        SNR(1,count) =20*log(energy/diff);
        count=count+1;
    end
%% Conclusion -> As we can see, the SNR increases as we quantize the signal more and more, so more information is lost
%% because of truncation. Thus when we take ratio between the 16-bit to 12-bit, the SNR will be more because more signal than
%% noise will be present, whereas, for the ratio between the original 16-bit and 1-bit, SNR wll be zero as there will be no signal left.

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% 6. High- Frequency Hearing Loss %%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function LPF = lpf(X, fs)
count=1;
    for fc=[500 1000 2000 4000]
        %fprintf('Filtering for cutoff_frequency %s\n', int2str(fc)); 
        Wn = (2/fs)*fc;
        b = fir1(20,Wn,'low',kaiser(21,3));
        fil = filter(b,1,X);
        %sound(fil,Fs{i})
        
        subplot(5,1,count);
        
        LPF = fftshift(fft(fil));
        df = fs/length(LPF);
        f = -fs/2:df:fs/2-df;
        plot(f,abs(LPF));
        xlabel('Frequency (in hertz)');
        ylabel(' Amplitude');
        title(strcat('Filtering with cutoff frequency = ',int2str(fc)));
        count=count+1;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% 7. Speech & Music Feature Extraction %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%% Percentage of “Low-Energy” Frames For Speech Signals%%%%%%%%%%%%%%%

function lenergyf = lefc(X,fs)
t=length(X(:,1))/fs;
audio = X(:,1);
N=2*t-1;
n=round(length(X)/t); %find how many samples will each frame contain
P=zeros(fs,N); %preallocate the matrix for 20 colums of Nsamples/20 in each
low_energy = zeros(1,N);
y=1;
count=0;

RMS1 = zeros(1,N);
    for k=0:N-1
        P(:,k+1)=audio(1+(n*k/2):n*((k+2)/2));
        RMS1(k+1) = rms(P(:,k+1));
    end

RMS10 = zeros(1,N*100);
P1 = zeros(fs/100,N*100);

    for j=1:N*100
        P1(:,j)= P((fs/100)*(j-1)+1:(fs/100)*j);
        RMS10(j) = rms(P1(:,j));
        if (RMS10(j) < RMS1(y))
            count=count+1;
        end
    
        if mod(j,100)==0
           low_energy(y)=count;
           count=0;
           y=y+1;
        end
    end
lenergyf = low_energy;
end

%%%% SpectralRolloff Point-> 95th Percentile PSD of each 1 second frame.%%%

function Percentile = sr(X,fs)
t=length(X(:,1))/fs;
audio = X(:,1);
N = 2*t-1;
n=round(length(X)/t); %find how many samples will each frame contain
P=zeros(fs,N);
for k=0:N-1
    P(:,k+1)=audio(1+(n*k/2):n*((k+2)/2));   
end
P1=P';
Percentile = zeros(1,N);
for m=1:N
    Q = periodogram(P1(m,:),hamming(length(P1(m,:))));
    Percentile(m) = prctile(Q,95);
end

end

%%%%%%%%%%%%%%%%%%%%%%% Spectral Centroid %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Spec = sc(X,fs)
t=length(X(:,1))/fs;
audio = X(:,1);
%audio =audio./rms(audio);
maxVec = max(audio);
minVec = min(audio);

%# normalize to -1...1
audio = ((audio-minVec)./(maxVec-minVec) - 0.5 ) *2;


N = 2*t-1;
n=round(length(X)/t); %find how many samples will each frame contain
P=zeros(fs,N);
    for k=0:N-1
        P(:,k+1)=audio(1+(n*k/2):n*((k+2)/2));   
    end
P1=P';
Spec = zeros(1,N);
    for m=1:N
        w_sum = 0; sum=0;
        [Sp,F] = periodogram(P1(m,:),hamming(length(P1(m,:))));
      
        %plot(F,10*log10(Pxx));
        for k=1:length(F)
            if Sp(k)~=0
                w_sum = w_sum + 10*log10(Sp(k))*F(k);
                sum = sum + 10*log10(Sp(k));
            end
        end
        Spec(m)= w_sum/sum;
    end
end

%%%%%%%%%% Spectral “Flux” (Delta Spectrum Magnitude): %%%%%%%%%%%%%%%%%%%

function SFlux = sf(X,fs)
t=length(X(:,1))/fs;
audio = X(:,1);
N = 2*t-1;
n=round(length(X)/t); %find how many samples will each frame contain
P=zeros(fs,N);
    for k=0:N-1
        P(:,k+1)=audio(1+(n*k/2):n*((k+2)/2));   
    end
P1=P';
SFlux = zeros(1,N);
    for m=1:N
        if (m>1)
            sdiff = P1(m,:) - P1(m-1,:);
            SFlux(m)= sqrt(sdiff * sdiff');
        end
    end
end

%%%%%%%%%%%%%%%%%%% Zero-Crossing Rate %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Zero_Cross = zc(X,fs)
t=length(X(:,1))/fs;
audio = X(:,1);
N = 2*t-1;
n=round(length(X)/t); %find how many samples will each frame contain
P=zeros(fs,N);
    for k=0:N-1
        P(:,k+1)=audio(1+(n*k/2):n*((k+2)/2));   
    end
P1=P';

Zero_Cross = zeros(1,N);
    for m=1:N
        num_zero = 0;
        for k=1:length(P1(m,:))
            if (k>1)
                if(((P1(m,k-1)>0) && (P1(m,k)<0))||((P1(m,k-1)<0) && (P1(m,k)>0)))
                    num_zero= num_zero +1;
                end
            end
        end
    Zero_Cross(m) = num_zero;      
    end
end