%% load dataset
clc;
clear all;
path_train = 'D:\ucsdCourse\2019WinterQuarter\ece271bSL2\hw\hw3\training set\';
path_train_image = strcat(path_train,'train-images.idx3-ubyte');
path_train_label = strcat(path_train,'train-labels.idx1-ubyte');
[train_imgs, train_labels] = readMNIST(path_train_image,path_train_label, 20000, 0);
assert(all(size(train_imgs)==[20000,784]));
assert(all(size(train_labels)==[20000,1]));
path_test = 'D:\ucsdCourse\2019WinterQuarter\ece271bSL2\hw\hw3\test set\';
path_test_image = strcat(path_test,'t10k-images.idx3-ubyte');
path_test_label = strcat(path_test,'t10k-labels.idx1-ubyte');
[test_imgs, test_labels] = readMNIST(path_test_image,path_test_label, 10000, 0);
assert(all(size(test_imgs)==[10000,784]));
assert(all(size(test_labels)==[10000,1]));


clearvars -except train_imgs test_imgs train_labels test_labels


%% adaboost 

[num_sample, dim] = size(train_imgs);
[num_test, ~] = size(test_imgs);

% class_i: 1~10
T = 250;
% 0 1 2 3 4 5 6 7 8 9
class_i = 9;
train_labels_i = -ones(size(train_labels));
train_labels_i(train_labels==class_i) = 1;
test_labels_i = -ones(size(test_labels));
test_labels_i(test_labels==class_i) = 1;

weights = zeros(num_sample,1);
g_train = zeros(num_sample,1);
g_test = zeros(num_test,1);

train_prob_error = zeros(T,1);
test_prob_error = zeros(T,1);

margin = zeros(num_sample,1);
store_margin = zeros(num_sample,5);
store_margin_count = 1;

max_weight_index = zeros(T,1);


for t=1:T
    margin = train_labels_i.*g_train;
    if ismember(t,[5,10,50,100,250])
        store_margin(:,store_margin_count) = margin;
        store_margin_count = store_margin_count +1 ;
    end
        
    weights = exp(-margin);
    
    [best_thresh_t,best_twin_t, best_index_t] = decision_stump(train_imgs, train_labels_i, weights);
    pred_train = alpha_t(train_imgs, best_thresh_t, best_twin_t, best_index_t);
    pred_test = alpha_t(test_imgs, best_thresh_t, best_twin_t, best_index_t);
    epsilon_t = sum(weights(pred_train~=train_labels_i))/sum(weights);
    Wt = 0.5*log((1-epsilon_t)/epsilon_t);
    
    g_train = g_train + Wt.*pred_train;
    g_test = g_test + Wt.*pred_test;
    
    g_pred_train = -ones(num_sample,1);
    g_pred_train(g_train>=0) = 1;
    
    g_pred_test = -ones(num_test,1);
    g_pred_test(g_test>=0) = 1;
    
    train_prob_error(t) = sum(g_pred_train~=train_labels_i)/num_sample;
    test_prob_error(t) = sum(g_pred_test~=test_labels_i)/num_test;
    [~,max_weight_index(t)] = max(weights);
    
end

% plot, save workspace

[num_per_index, which_index] = hist(max_weight_index,unique(max_weight_index));
[~, sort_index] = sort(-num_per_index);
freq_index = which_index(sort_index);
three_most_common_index = freq_index(1:3);
for i = 1:3
    figure;
    imshow(reshape(train_imgs(three_most_common_index(i),:),28,28)');
end

filename = sprintf('digit%d.mat',class_i);
save(filename);
figure;
plot(1:T,train_prob_error,1:T,test_prob_error);
%plot(1:T,train_prob_error);
xlabel('number of iterations');
ylabel('probability error');
title(['digit',num2str(class_i)]);
legend('train','test');

figure;
plot(1:T,max_weight_index);
xlabel('number of iteration');
ylabel('index of largest weight');
title(['digit',num2str(class_i)]);

figure;

cdfplot(store_margin(:,1));
hold on;
cdfplot(store_margin(:,2));
hold on;
cdfplot(store_margin(:,3));
hold on;
cdfplot(store_margin(:,4));
hold on;
cdfplot(store_margin(:,5));
hold off;
xlabel('margin')
ylabel('cumulative distribution');
legend('5th iter','10th iter','50th iter','100th iter','250ith iter');
title(['digit',num2str(class_i)]);



%% function 
% argmax 
function [best_thresh, best_twin, best_index] = decision_stump(x, y, weights)
    thresholds = (0:50)/50;
    [num_sample, dim] = size(x);

    thresh_dim = zeros(dim,1);
    twin_dim = zeros(dim,1);
    weight_pred_y_dim = zeros(dim,1);
    
    for i = 1:dim
        x_ith = x(:,i);
        best_thresh_j = -1;
        best_twin_j = 0;
        max_weight_pred_y = -inf;
        for j = 1:51
            pred = -ones(num_sample,1);
            pred(x_ith>=thresholds(j)) = 1;
           
            if sum(pred==y) < num_sample/2
                twin = -1;
                pred = -pred;
            else
                twin = 1;
                
            end
            weight_pred_y = sum(weights.*pred.*y);
            if weight_pred_y >= max_weight_pred_y
                max_weight_pred_y = weight_pred_y;
                best_thresh_j = thresholds(j);
                best_twin_j = twin;
            end
        end
        thresh_dim(i) = best_thresh_j;
        twin_dim(i) = best_twin_j;
        weight_pred_y_dim(i) = max_weight_pred_y;
        
    end
    [~, best_index] = max(weight_pred_y_dim);
    best_thresh = thresh_dim(best_index);
    best_twin = twin_dim(best_index);
end

function pred = alpha_t(x,thresh,twin,i_dim)
    [num_sample,~] = size(x);
    x_ith = x(:,i_dim);
    pred = -ones(num_sample,1);
    pred((x_ith>=thresh)) = 1;
    pred = pred*twin;
end


