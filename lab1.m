
%{

%% Problem 1: Simulering av exponentialfordelade slumptal
mu = 10;
N = 1e4;
y = exprnd(mu, N, 1); % Genererar N exp-slumptal
hist_density(y); % Skapar ett normaliserat histogram
t = linspace(0, 100, N/10);% Vektor med N/10 punkter
figure(1);
hold on
plot(t, exppdf(t, mu), 'r') % 'r' betyder rod linje
hold off

%relativit bra ju fler slumptal desto närmare är vi den röda linjen

%% Problem 2: Stora talens lag
mu = 0.5;
M = 500; %around 1k is perfect sometimes good though and sometimes bad
X = exprnd(mu, M, 1);
figure(1);
plot(ones(M, 1)*mu, 'r-.')
hold on
for k = 1:M
plot(k, mean(X(1:k)), '.')
if k == 1
 legend('Sant \mu', 'Skattning av ...\mu','AutoUpdate','off')
 end
 xlabel(num2str(k))
end



%% Problem 3: Vantevarde av exp.fordelad stokastisk variabel
mu= 10;
N = 1e5;
y = exprnd(mu, N, 1);
mean(y)
%Ganska bra för stora N enligt stora lagens lag går vi mot my



%% Problem 4: Monte Carlo-skattning av talet pi
N = 1e2;
U = 2*rand(1,N)-1; % Genererar U(-1,1)-ford. slumptal
V = 2*rand(1,N)-1;
plot(U,V,'o'), hold on % Plottar de genererade punkterna
X = -1:0.01:1;
plot(X,sqrt(1-X.^2),'r') % Plottar enhetscirkeln
plot(X,-sqrt(1-X.^2),'r')
Z = (sqrt(U.^2+V.^2)<= 1); % Beraknar narmevarde pa pi
pi = 4*mean(Z)

%%problem 5
p1k1 = binocdf(3,10,0.3)
p2k1 = binocdf(7,10,0.3, 'upper')
p3k1 = binocdf(4,10,0.3) - binocdf(4,10,0.3);

p1k2 = normcdf(3,5,3)
p2k2 = normcdf(7,5,3, 'upper')
p3k2 = normcdf(4,5,3) - normcdf(3,5,3);

p1k3 = expcdf(3,7)
p2k3 = expcdf(7,7, 'upper')
p3k3 = expcdf(4,7) - normcdf(3,7);


%% Problem 6: Tathetsfunktion for normalfordelning
dx = 0.01;
x = -10:dx:10; % Skapar en vektor med dx som inkrement
y = normpdf(x,2,2);

%plot(x,y)


%% Problem 6: Tathetsfunktion for gammafordelning
dx = 0.01;
x = 0:dx:10; % Skapar en vektor med dx som inkrement
y = gampdf(x,1,2);
figure(1);
plot(x,y), hold on
z = gampdf(x,5,1);
plot(x,z,'r')


%% Problem 6: Fordelningsfunktion for gammafordelning
dx = 0.01;
x = 0:dx:10; % Skapar en vektor med dx som inkrement
y = gamcdf(x,1,2);
figure(1);
plot(x,y), hold on
z = gamcdf(x,5,1);
plot(x,z,'r')

%}

%% Problem 7: Multivariat normalfordelning
mux = 0; muy = 5; sigmax = 5; sigmay = 4; rho = 0.7;
figure(1);
plot_mvnpdf(mux, muy, sigmax, sigmay, rho)

%ju mindre sigma i båda desto mer koncentrerat har vi det kring (myx, myy)
%desto större sigmas leder till bredare fördelning. my parametrerna är det
%som bestämmer centrum för fördelningen (x,y).