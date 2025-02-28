function[] = generate_xgb_data()
prefix1 = 'mouse5';
prefix2 = 'evt22';  
load("C:\Users\kasra\Desktop\XGBOOST\mouse5\evt_rec\evt_rec22.mat");
load("K:\2Dup\Qian_lablled_videos\mouse5\mouse5_hf_2024-06-23-143824-0000\SSM120_data.mat");
load("C:\Users\kasra\Desktop\XGBOOST\mouse5\spike_data.mat")
evt = evt(2:end);
Nevt = numel(evt);

%remove all spike before after events
dt = median(diff(evt));
ind = find((spk>=evt(1)-dt)&(spk<=evt(end)+dt));
spk = spk(ind);
clu = clu(ind);

%generate output (Y) for xgb
clu_val = unique(clu);
Ncell = numel(clu_val);
Y = zeros(Nevt,Ncell);
for n = 1:Ncell
    Y(:,n) = hist(spk(clu==clu_val(n)),evt)';
end

%remove cells with few spikes
TH = 100;
sum_spk = sum(Y);
ind_ok = find(sum_spk>TH);
Y = Y(:,ind_ok);
clu_val = clu_val(ind_ok);

%generate input (X) for xgb
X = [b' dist' [1:Nevt]'];
Nd = size(X,2);

%add shuffle controls
Nsh = 9;
X  = repmat(X,1,1,Nsh+1);
for n = 1:Nsh
    for m = 1:Nd-1
        X(:,m,n+1) = X(randperm(Nevt),m,n+1);
    end
end

%generate blocks for cross-validation
K = 5;
Lblock = 150;
which_fold = [];
N = 0;
while N <= Nevt
      fold_val = randperm(K);  
      for n = 1:K
          which_fold = [which_fold fold_val(n)*ones(1,Lblock)];  
      end
      N = numel(which_fold);
end
which_fold = which_fold(1:Nevt);
    
% Construct the final filename with prefixes
save_name = strcat(prefix1, '_', prefix2, '_', 'data_for_prediction.mat');
    
%save
save(save_name,'X','Y','K','which_fold','clu_val');