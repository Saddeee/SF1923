
 


%% Problem 1: Simulering av konfidensintervall

%{
% Parametrar:
n = 25; %Antal matningar
mu = 10; %Vantevardet
sigma = 1; %Standardavvikelsen
alpha = 0.05;
%Simulerar n observationer for varje intervall
x = normrnd(mu, sigma,n,100); %n x 100 matris med varden
%Skattar mu med medelvardet
xbar = mean(x); %vektor med 100 medelvarden.
%Beraknar de undre och ovre granserna
undre = xbar - norminv(1-alpha/2)*sigma/sqrt(n)
ovre = xbar + norminv(1-alpha/2)*sigma/sqrt(n)

p=0;
for i= 1:length(ovre)
    if undre(i) <= mu && mu <= ovre(i)
        p=p+1;
    end
end

disp(p/100)

%Hur många av dessa intervall kan förväntas innehålla det sanna värdet på 
% µ?
%med stor sannolikhet ligger mu i intervallet efter 100 försök
%jag tror det fanns en sats som sa att när n-> infite går vi mot
%konfidensgraden dvs 95% => 100 - 5 st?

% Problem 1: Simulering av konfidensintervall (forts.)
%Ritar upp alla intervall
figure(1)
hold on
    for k=1:100
        if ovre(k) < mu % Rodmarkerar intervall som missar mu
            plot([undre(k) ovre(k)],[k k],'r')
            elseif undre(k) > mu
            plot([undre(k) ovre(k)],[k k],'r')
        else
            plot([undre(k) ovre(k)],[k k],'b')
        end
    end
%b1 och b2 ar bara till for att figuren ska se snygg ut.
b1 = min(xbar - norminv(1 - alpha/2)*sigma/sqrt(n));
b2 = max(xbar + norminv(1 - alpha/2)*sigma/sqrt(n));
axis([b1 b2 0 101]) %Tar bort outnyttjat utrymme i figuren
%Ritar ut det sanna vardet
plot([mu mu],[0 101],'g')
hold off

%om vi bara ändrar mu så ändras inte resultatet sådär jätte mycket då vi
%bara förflyttar normalfördelningen. Om vi ändrar och gör den mycket större
%hamnar nästan allt utanför intervallet. Om vi gör den 10 gånger större
%sjunker sannolikheten att den är i intervallet efter 100 försök. till
%ungefär 50%. Om vi ökar antalet n så ser vi att vi förbättrar
%sannolikehten och överenstämmer med sats.
%}

%{
% Problem 2: Maximum likelihood/Minsta kvadrat
M = 1e4;
b = 4;
x = raylrnd(b, M, 1);
hist_density(x, 40)
figure(1);
hold on
my_est_ml = (1/(2*M)*sum(x.^2))^0.5;% Skriv in din ML-skattning har
my_est_mk = mean(x)*(2/pi)^0.5;% Skriv in din MK-skattning har
plot(my_est_ml, 0, 'r*')
 plot(my_est_mk, 0, 'g*')
 plot(b, 0, 'ro')
 plot(0:0.1:6, raylpdf(0:0.1:6, my_est_ml), 'r')
 plot(0:0.1:6, raylpdf(0:0.1:6, my_est_mk), 'b')
 hold off
%skattning ser väldigt bra ut.

%}

%{
% Problem 3: Konfidensintervall for Rayleighfordelning
load wave_data.mat
alfa=0.05;
figure(1);
my_est = mean(y)*(2/pi)^0.5;% Skriv in din MK-skattning har
subplot(2,1,1), plot(y(1:end))
subplot(2,1,2), hist_density(y)

% Problem 3: Konfidensintervall (forts.)
standardav = (2/pi * 1/(length(y)) *(2-pi/2)*my_est^2)^0.5;
% standardav = ( (4 - pi)*my_est) / (pi * (length(y)^0.5));
interval = (my_est + tinv([alfa, (1-alfa)], length(y)-1)*standardav);
lower_bound = interval(1,1)
upper_bound = interval(1,2)
hold on % Gor sa att ploten halls kvar
plot(lower_bound, 0, 'g*')
plot(upper_bound, 0, 'g*')
plot(0:0.1:6, raylpdf(0:0.1:6, my_est), 'r')
hold off

%fördelnign ser bra ut som svar på problem 3
%}
%{

load birth.dat
x= birth(birth(:, 20) < 3, 3);
y = birth(birth(:,20) == 3, 3);

length(x)
length(y)

% Problem 4: Fordelningar av givna data (forts.)
figure(1);
subplot(2,2,1), boxplot(x),
axis([0 2 500 5000])
subplot(2,2,2), boxplot(y),
axis([0 2 500 5000])


% Problem 4: Fordelningar av givna data (forts.)
subplot(2,2,3:4), ksdensity(x),
hold on
[fy, ty] = ksdensity(y);
plot(ty, fy, 'r')
hold off

%det vi kan dra för slutsatser är att den som inte röker är mer förskjuten
%åt höger dvs löper mertalet hamnar över gränsen som låg vikt på 2500
%däremot ser man dock att de som röker ligger lite till vänster med sitt
%värde och vi kan också se att de är approximativt normalfördelat. De som
%röker har mer koncentrerade på en viss vikt löper större sannolikhet att
%hamna där. Dock de som inte röker har en bredarea standardavvikelse.

%intreesant att veta om alkohol påverkar




%%problem 4 eget med alkhol


load birth.dat
not_alko = birth(birth(:, 26) < 2, 3);
alko = birth(birth(:,26) == 2, 3);

% Problem 4: Fordelningar av givna data (forts.)
figure(2);
subplot(2,2,1), boxplot(not_alko),
axis([0 2 500 5000])
subplot(2,2,2), boxplot(alko),
axis([0 2 500 5000])


% Problem 4: Fordelningar av givna data (forts.)
subplot(2,2,3:4), ksdensity(not_alko),
hold on
[fy, ty] = ksdensity(alko);
plot(ty, fy, '-')
hold off

%vi ser från plotten likt förra att de som dricker är mer koncetrerat i
%mitten större sannolikhet att hamna i en okej vikt på 3500. Men vi ser att
%den är dock lite mer förskjuten än de som inte dricker till vänster dvs
%mot den lägre vikten. Allstå det finns en risk för vissa som dricker att
%deras barn faktiskt blir underviktiga mer än den som inte dricker iaf. Å
%andra sidan ser vi att de som inte dricker är mer försjuktna åt höger men
%också större standardavvikelse men sannolikhet att du hamnar på en okej
%viktig är större pga försjutknignen. Men det finns såklart risk att man
%kan hamna i undervittiga delen men lägre än de som dricker.


%



% problem 5

age = birth(:, 4)
figure(3);
subplot(2,2,3:4), qqplot(age)



% Weight of mother
w = birth(:, 15);
figure(4);
subplot(2,2,3:4), qqplot(w)


[h_age, p_age] = jbtest(age, 0.05);
[h_w,   p_w]   = jbtest(w,   0.05);
%vi förkasar att age och weight när normalfördenalde
fprintf('Age:    JB h = %d, p = %.4f\n', h_age, p_age)
fprintf('Weight: JB h = %d, p = %.4f\n', h_w,   p_w)

%}

%{
% Seems like age is not normnally ditributed

% Problem 6

alfa = 0.05;
figure(5);
good_parent = birth(birth(:, 20) < 3, 3);
bad_parent = birth(birth(:, 20) == 3, 3);

mean_diff = mean(good_parent) - mean(bad_parent); % Mean difference in expected length stickprovsmedel

%standardav = (std(bad_parent)^2 / length(bad_parent) + std(good_parent)^2 / length(good_parent))^0.5;

%om vi antar samma std
viktadstandardav = ((((length(good_parent)-1)*std(good_parent)^2+(length(bad_parent)-1)*std(bad_parent)^2))/(length(good_parent)+length(bad_parent)-2));
stdest= viktadstandardav*sqrt((1/(length(good_parent)+1/(length(bad_parent)))));

freedom= length(good_parent)+length(bad_parent)-2;
interval = (mean_diff + tinv([alfa, (1-alfa)], freedom)*standardav)
lower_bound = interval(1,1);
upper_bound = interval(1,2);
hold on % Gor sa att ploten halls kvar
plot(lower_bound, 0, 'g*')
plot(mean_diff, 0, 'r*')
plot(upper_bound, 0, 'g*')

% Alltså skillanden är typ 144.6744 med konfidensintervall 
% 67.3471 till 222.0017, 
% Vi vet alltså med säkerhet att barn med föräldrar som inte röker väger
% något mer, dryga 144 g mer

%}


load moore.dat

figure(6)
y = moore(:, 2);
length(y);

w = log(y);
x = moore(:, 1);
x_matrix = [ones(length(x), 1), x];
%ksdensity(w)
%ksdensity(logged)


beta_hat = regress(w, x_matrix)
y_tmp = x_matrix*beta_hat;
plot(x,y_tmp)
% Problem 6: Regression
res = w-x_matrix*beta_hat;
subplot(2,1,1), normplot(res)
subplot(2,1,2), hist(res)

% NormalFördelning?
pred2025 =exp([1, 2025]*beta_hat);
fprintf('predict transis 2025 is %d', pred2025);

%2025 finns det: 1.359867e+08

