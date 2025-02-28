function [] = cluster_with_optimal_k()

    % Ask the user to input the best K
    K = input('Enter the optimal number of clusters K based on the plots: ');
    % Load data for the chosen K
    load(sprintf('Clustering_Dataset_K%d.mat', K), 'score_train', 'score_test', 'group_train', 'group_test', 'Y_train', 'Y_test', 'C');

    % Stats
    [diff_hist_train, hist_mel_bright_train, hist_mel_dark_train] = compare_dark_bright(group_train, Y_train, 1:K);
    [diff_hist_test, hist_mel_bright_test, hist_mel_dark_test] = compare_dark_bright(group_test, Y_test, 1:K);

    % Shuffle stats for significance testing
    Nshuf = 10000;
    diff_hist_shuf = zeros(Nshuf, K);
    for n = 1:Nshuf
        diff_hist_shuf(n,:) = compare_dark_bright(group_test, Y_test(randperm(length(Y_test))), 1:K); 
    end

    % Figures
    figure; hold on;
    bar(diff_hist_test);
    plot(diff_hist_shuf','.K');

    % Save results
    save('Final_Clustering_Dataset.mat', 'score_train', 'score_test', 'group_train', 'group_test', 'Y_train', 'Y_test', 'C');
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