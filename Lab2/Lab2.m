clear;
close all;

% I часть

x = @ (t) sin(-2.*sin(t).*t.^2 + t.*7);
t = 0.5:0.01:3.2;

P = x(t);

plot(t, P, '.-');
grid;

D=5; % Глубина

Pi = zeros(1,D);
PM = zeros(1,D);

% Задаём последовательность
for i=1:D
    Pi(i) = P(i);
end

for i=1:size(P,2) - D
    PM(i) = P(i + D);
end

net = newlin([0, D], 1, [1 2 3 4 5], 0.01); % Input from x to y, number of outputs, delays, learning rate

display(net);
view(net);

% Обучение
net.inputWeights{1, 1}.initFcn = 'rands'; % Инициализируем веса и смещения случайными значениями
net.biases{1}.initFcn = 'rands';
net = init(net);

IW = net.IW{1, 1};
b = net.b{1};

M1 = sqrt(mse(PM - net(PM)));

Pi = con2seq(Pi);
PM = con2seq(PM);
P = con2seq(P);

net.adaptParam.passes = 50;

[net, ~, ~] = adapt(net, PM, PM, Pi);

Y = sim(net, P, Pi);

Y = seq2con(Y);
Y = Y{1};
P = seq2con(P);
P = P{1};
E = Y - P;

M2 = sqrt(mse(Y - P));

first_part = figure;
subplot(211);
plot(t, P, 'b', t, Y, 'r--')
grid;

subplot(212);
plot(t, E, 'g');
grid;

uiwait(first_part);

clear;
close all;

% II часть

x = @ (t) sin(-2.*sin(t).*t.^2 + t.*7);
t = 0.5:0.01:3.2;

P = x(t);

plot(t, P, '.-');
grid;

D=3; % Глубина

Pi = zeros(1,D);
PM = zeros(1,D); % Вход
PM1 = zeros(1,D); % Выход

% Задаём последовательность
for i=1:D
    Pi(i) = P(i);
end

for i=1:size(P,2) - D
    PM(i) = P(i + D - 1);
end

for i=1:size(P,2) - D
    PM1(i) = P(i + D);
end

P = con2seq(P);
learning_rate = maxlinlr(cell2mat(P), 'bias');

net = newlin([-1, 1], [-1, 1], [1 2 3], learning_rate);

% Обучение
net.inputWeights{1, 1}.initFcn = 'rands'; % Инициализируем веса и смещения случайными значениями
net.biases{1}.initFcn = 'rands';
net = init(net);

display(net);
view(net);

IW = net.IW{1, 1};
b = net.b{1};

M1 = sqrt(mse(PM - net(PM)));

Pi = con2seq(Pi);
PM = con2seq(PM);
PM1 = con2seq(PM1);

net.trainParam.goal = 1e-6;
net.trainParam.epochs = 600;
net = train(net, PM, PM1, Pi);

Y = sim(net, P, Pi);

Y = seq2con(Y);
Y = Y{1};
P = seq2con(P);
P = P{1};
E = Y - P;

M2 = sqrt(mse(Y - P));

figure;
subplot(211);
plot(t, P, 'b', t, Y, 'r--')
grid;

subplot(212);
plot(t, E, 'g');
grid;

% Продлеваем
t1 = 0.5:0.01:3.3;
P1 = x(t1);
P2 = con2seq(P1);

Y1 = sim(net, P2);

Y1 = seq2con(Y1);
Y1 = Y1{1};
P2 = seq2con(P2);
P2 = P2{1};
E1 = Y1 - P2;

M3 = sqrt(mse(Y1 - P2));

second_part = figure;
subplot(211);
plot(t1, P2, 'b', t1, Y1, 'r--')
grid;

subplot(212);
plot(t1, E1, 'g');
grid;

uiwait(second_part);

clear;
close all;

% III часть

x = @ (t) cos(-cos(t).*t.^2 + t);
t = 0.5:0.01:4;

y = @(t) 1/4.*(cos(-cos(t).*t.^2 + t + pi.*2));

P = x(t);
T = y(t);

plot(t, P, '.-', t, T, '-');
grid;

D = 4;

P2 = zeros(D,size(P, 1));
P3 = zeros(D,size(P, 1));

for i=1:size(P,2)
    P2(i + D - 1) = P(i);
end

for i=1:D
    P3(i,1:size(P, 2)) = P2(i:size(P, 2) + i - 1);
end

T = con2seq(T);
P = con2seq(P3);

net = newlind(P, T);

Y = sim(net, P);

Y = seq2con(Y);
Y = Y{1};
T = seq2con(T);
T = T{1};

M = sqrt(mse(T - Y));

figure;
plot(t, T, 'b', t, Y, 'r--');
grid;




