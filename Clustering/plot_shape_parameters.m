function[] = cluster_behaviour_kasra()
    rng(1)
    filepath = 'K:\qian_behav_cluster\preference_SSM_Q120';
    Nmice = 120; Nsample = 15; Nshuf = 10000;
    mel_dark = readcell('K:\qian_behav_cluster\preference_SSM_Q120\MICELISTQ1-Q120-pref.xlsx');
    mel_dark = vertcat(mel_dark{2:end,4});

    % Collect training set
    X = []; Y = []; mouse_id = [];
    for n = 1:Nmice
        load([filepath '\Q' num2str(n) '\SSM' num2str(n) '_data.mat'],'b','dist');
        load([filepath '\Q' num2str(n) '\preference_Q' num2str(n) '.mat'],'preference');
        Nt = numel(preference); 
        if mel_dark == 'R'
           preference = ~preference;
        end
        Npoint = floor(Nt/Nsample);
        for m = 1:Npoint
            temp = b(:,(m-1)*Nsample+1:m*Nsample);
            X = [X; temp(:)' dist((m-1)*Nsample+1:m*Nsample')];
            mouse_id(m) = n;
            Y =[Y;  mean(preference((m-1)*Nsample+1:m*Nsample))];
        end
        disp(sprintf('Mouse %s',num2str(n)));
    end
    [L,Nd] = size(X);

    % PCA 
    for n = 1:Nd
        X(:,n) = (X(:,n)-mean(X(:,n)))/std(X(:,n));
    end
    [coeff,score,lambda] = pca(X);
    idx = min(find(cumsum(lambda)/sum(lambda)>0.9));
    score = score(:,1:idx);

    % Determine the best K using the Elbow method
    maxK = 90;  % Set the maximum number of clusters to 90
    sumd = zeros(1,maxK);
    for k = 1:maxK
        [~,~,sumd(k)] = kmeans(score, k, 'Distance', 'correlation', 'Replicates', 100, 'MaxIter', 800);
    end
    
    % Plot the Elbow curve
    figure;
    plot(1:maxK, sumd, 'bx-');
    xlabel('Number of clusters K');
    ylabel('Sum of within-cluster variance');
    title('Elbow Method for Optimal K');
    
    % Ask the user to input the best K
    K = input('Enter the optimal number of clusters K based on the Elbow plot: ');

    % Clustering with the chosen K
    [group,C] = kmeans(score, K, 'Distance', 'correlation', 'Replicates', 100, 'MaxIter', 800);
    group_val = 1:K;

    % Stats
    [diff_hist,hist_mel_bright,hist_mel_dark] = compare_dark_bright(group,Y,group_val);
    for n = 1:Nshuf
        diff_hist_shuf(n,:) = compare_dark_bright(group,Y(randperm(L)),group_val); 
    end

    % Figure
    figure; hold on;
    bar(diff_hist);
    plot(diff_hist_shuf','.k');

    % Save
    save('Clustering_Dataset');

    % Decode TODO

    function[diff_hist,hist_mel_bright,hist_mel_dark] = compare_dark_bright(group,Y,group_val)
        hist_mel_bright = hist(group(Y<0.5), group_val) / sum(Y<0.5);
        hist_mel_dark = hist(group(Y>0.5), group_val) / sum(Y>0.5);
        diff_hist = hist_mel_bright - hist_mel_dark;
    end
end
