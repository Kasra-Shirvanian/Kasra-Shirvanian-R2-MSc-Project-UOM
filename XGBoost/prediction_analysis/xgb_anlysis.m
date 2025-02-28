load("C:\Users\kasra\Desktop\XGBOOST\mouse3_evt8_data_for_prediction.mat");
load("C:\Users\kasra\Desktop\XGBOOST\mouse3_evt8_data_for_prediction_results.mat");
load("C:\Users\kasra\Desktop\XGBOOST\mouse3\spike_data.mat")

[Nt,Ncell,Nsh] = size(Ypred);
Nsh = Nsh-1;

C0 = zeros(1,Ncell);
THsig = zeros(1,Ncell);
Csh = zeros(1,Ncell);

for n = 1:Ncell
    C0(n) = corr(Y(:,n),squeeze(Ypred(:,n,1)));
    Ctemp = zeros(1,Nsh);
    for m = 1:Nsh
        Ctemp(m) = corr(Y(:,n),squeeze(Ypred(:,n,m+1)));
    end
    Csh(n) = mean(Ctemp);
    THsig(n) = mean(Ctemp)+5*std(Ctemp);
end
ind_signif = find(C0>THsig);


figure; hold on;
plot(Csh,C0,'.','MarkerSize',20);
plot(Csh(ind_signif),C0(ind_signif),'o','LineWidth',2);
line(1.1*[0 max(C0)],1.1*[0 max(C0)]);
xlabel('Corr_{shuf}');
ylabel('Corr_{data}')

%n = 10;
%neu_pred = poissrnd(Ypred(:,ind_signif(n),1));
%neu = Y(:,ind_signif(n));
%pcolor(hist3([neu neu_pred],{0:max(neu), 0:max(neu)}))
figure;
plot(filter(ones(1,150)/150,1,Ypred(:,ind_signif(1),1)))
hold on
plot(filter(ones(1,150)/150,1,Y(:,ind_signif(1))))
ind_signif = find(C0>THsig)
xlabel('Frames');
ylabel('Spike_Count')
