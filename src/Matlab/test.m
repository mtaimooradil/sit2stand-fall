Fs = 200;
Ts = 1/Fs;
t = 0:Ts:1;
F = 10;
x = sin(2*pi*F*t);
noise = 1 * rand(size(t))-0.5;
x_noisy = x + noise;
subplot(2,1,1); plot(t,x_noisy);xlabel('time');ylabel('Amplitude');

%autocorr(x_noisy);

[Rxx, lags] = xcov(x_noisy, x_noisy);
subplot(2,1,2);plot(lags, Rxx);xlabel('time');ylabel('Acr');

x = y(1:400);%randn(100,1);
x = x-mean(x);
%x = 1:3;
% y = acc;%sin(2*pi*5*x);%filter([1 -1 1],1,x);
% y=y-mean(y);
[c,l] = xcorr(x,x,'coeff');
[C,L] = xcov(x,x,'coeff');
[acf,lags,bounds] = autocorr(x,399);

subplot(311);
stem(l(51:end),c(51:end)); title('xcorr');
subplot(312)
stem(L(51:end),C(51:end)); title('xcov');
subplot(313)
stem(lags,acf); title('autocorr')

Fs = 106;            % Sampling frequency                    
T = 1/Fs;             % Sampling period       
L = 2000;             % Length of signal
t = (0:L-1)*T;        % Time vector

S = sin(2*pi*8*t) + sin(2*pi*12*t);
signal = sin(2*pi*3.816*t) + sin(2*pi*7.632*t) + sin(2*pi*11.554*t) + sin(2*pi*15.37*t);

L = length(filt_acc);
Y = fft(filt_acc);
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);

f = Fs*(0:(L/2))/L;
plot(f,P1) 
title("Single-Sided Amplitude Spectrum of X(t)")
xlabel("f (Hz)")
ylabel("|P1(f)|")


fc = 10; % cutoff frequency [Hz]
fs = 106; % Sampling Frequency [Hz]
n_order = 4; % filter order

[b, a] = butter(n_order, fc/(fs/2));
filt_acc = filtfilt(b, a, my_acc_4);

count = 0;
transition_indices = [];
for i=1:length(mult)
    if mult(i) < 0
        if filt_acc_2(i) > 0 % from positive to negative
            count = count + 1;
            transition_points(i) = filt_acc_2(i);
            transition_indices = [transition_indices,i];
        else
            transition_points(i) = nan;
        end
    else
        transition_points(i) = nan;
    end
end

max_indices = [];
for i=1:6%length(transition_indices)-1
    arr = filt_acc_2(transition_indices(i): transition_indices(i+1));
    [m, idx] = max(arr);
    disp(transition_indices(i))
    disp(idx + transition_indices(i) - 1)
    disp(m)
    max_indices = [max_indices,idx + transition_indices(i) - 1];
end
