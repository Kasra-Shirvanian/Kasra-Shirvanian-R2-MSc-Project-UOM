function [] = find_optimal_k()
    rng(1)
    filepath = 'K:\qian_behav_cluster\preference_SSM_Q120';
    Nmice = 20; Nsample = 15; Nshuf = 10000;
    mel_dark = readcell('K:\qian_behav_cluster\preference_SSM_Q120\MICELISTQ1-Q120-pref.xlsx');
    mel_dark = vertcat(mel_dark{2:end,4});

    % Dynamic range for time spent in the left chamber
    range = sprintf('E2:E%d', Nmice + 1);
    time_spent_left = cell2mat(readcell('K:\qian_behav_cluster\preference_SSM_Q120\MICELISTQ1-Q120-pref.xlsx', 'Range', range));

    % Translate time spent in left chamber to time spent in dark chamber
    time_spent_dark = time_spent_left;
    for n = 1:Nmice
        if mel_dark(n) == 'R'
            time_spent_dark(n) = 1 - time_spent_left(n);
        end
    end

    % Collect dataset
    X = []; Y = []; mouse_id = [];
    for n = 1:Nmice
        load([filepath '\Q' num2str(n) '\SSM' num2str(n) '_data.mat'],'b','dist');
        load([filepath '\Q' num2str(n) '\preference_Q' num2str(n) '.mat'],'preference');
        Nt = numel(preference); 
        if mel_dark(n)== 'R'
            preference = ~preference;
        end
        Npoint = floor(Nt/Nsample);
        for m = 1:Npoint
            temp = b(:,(m-1)*Nsample+1:m*Nsample)';
            X = [X; temp(:)' dist((m-1)*Nsample+1:m*Nsample')];
            mouse_id = [mouse_id; n];
            Y =[Y;  mean(preference((m-1)*Nsample+1:m*Nsample))];
        end
        disp(sprintf('Mouse %s',num2str(n)));
    end
    [L,Nd] = size(X);

    % Split data into training and testing sets based on mice
    cv = cvpartition(Nmice, 'HoldOut', 0.5);
    trainMice = find(training(cv));
    testMice = find(test(cv));

    % Calculate average time spent in the dark chamber for training and test sets
    avg_time_dark_train = mean(time_spent_dark(trainMice));
    avg_time_dark_test = mean(time_spent_dark(testMice));

    % Create logical indices for training and testing samples based on mice
    trainIdx = ismember(mouse_id, trainMice);
    testIdx = ismember(mouse_id, testMice);

    % Display indices of training and test sets
    disp('Training set mice:');
    disp(trainMice);
    disp('Test set mice:');
    disp(testMice);

    X_train = X(trainIdx, :);
    Y_train = Y(trainIdx, :);
    X_test = X(testIdx, :);
    Y_test = Y(testIdx, :);

    disp(size(X_train));

    % Standardize features
    m = mean(X_train); s = std(X_train);
    for n = 1:Nd
        X_train(:,n) = (X_train(:,n)-m(n))/s(n);
        X_test(:,n) = (X_test(:,n)-m(n))/s(n); % Use train mean and std for test set
    end

    % PCA on training set
    [coeff,score_train,lambda] = pca(X_train);
    idx = min(find(cumsum(lambda)/sum(lambda)>0.9));
    score_train = score_train(:,1:idx);

    % Project test data onto the same PCA space
    score_test = X_test * coeff(:,1:idx);

    % Determine the best K using the Elbow method
    maxK = 20;  % Set the maximum number of clusters
    wcss = zeros(1,maxK);
    correctness_train = zeros(1,maxK);
    correctness_test = zeros(1,maxK);

    for K = 1:maxK
        [group_train,C,sumd] = kmeans(score_train, K, 'Distance', 'correlation', 'Replicates', 100, 'MaxIter', 1000);
        group_test = knnsearch(C, score_test, 'Distance', 'correlation');

        % Evaluate correctness for training set
        clusters_bias = zeros(1, K);
        for cl = 1:K
            group_indices = (group_train == cl);
            clusters_bias(cl) = mean(Y_train(group_indices));
        end

        correctness_train(K) = evaluate_correctness(group_train, Y_train, K, clusters_bias, avg_time_dark_train);
        
        % Evaluate correctness for test set
        correctness_test(K) = evaluate_correctness(group_test, Y_test, K, clusters_bias, avg_time_dark_test);

        % Within-cluster sum of squares for Elbow method
        wcss(K) = sum(sumd);

        % Save data for each K
        save(sprintf('Clustering_Dataset_K%d.mat', K), 'score_train', 'score_test', 'group_train', 'group_test', 'Y_train', 'Y_test', 'C');
    end

    % Plot the Elbow curve
    figure;
    subplot(2,1,1);
    plot(1:maxK, wcss, 'bx-');
    xlabel('Number of clusters K');
    ylabel('Sum of within-cluster variance');
    title('Elbow Method for Optimal K');

    % Plot correctness
    subplot(2,1,2);
    plot(1:maxK, correctness_train, 'r-', 1:maxK, correctness_test, 'b-');
    legend('Training set', 'Test set');
    xlabel('Number of clusters K');
    ylabel('Correctness');
    title('Correctness for Different K values');
end

% Function to evaluate correctness
function correctness = evaluate_correctness(groups, preferences, K, clusters_bias, avg_time_dark)
    total_correct_frames = 0;
    total_frames = numel(preferences);
    
    for cl = 1:K
        group_indices = (groups == cl);
        if clusters_bias(cl) > avg_time_dark
            total_correct_frames = total_correct_frames + sum(preferences(group_indices) > avg_time_dark);
        else
            total_correct_frames = total_correct_frames + sum(preferences(group_indices) <= avg_time_dark);
        end
    end
    
    correctness = total_correct_frames / total_frames;
end

% Function to compare dark and bright
function [diff_hist, hist_mel_bright, hist_mel_dark] = compare_dark_bright(group, Y, group_val)
    hist_mel_bright = hist(group(Y<0.5), group_val) / sum(Y<0.5);
    hist_mel_dark = hist(group(Y>0.5), group_val) / sum(Y>0.5);
    diff_hist = hist_mel_bright - hist_mel_dark;
end

% Function to assign clusters
function idx = assign_clusters(score, C)
    K = size(C,1);
    N = size(score,1);
    idx = zeros(N,1);
    for n = 1:N
        dist = sum((C - score(n,:)).^2, 2);
        [~, idx(n)] = min(dist);
    end
end
