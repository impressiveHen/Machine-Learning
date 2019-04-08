all_g_test = zeros(10000,10);
all_g_train = zeros(20000,10);
for class_number = 0:9
    file = sprintf('digit%d.mat',class_number);
    load(file);
    all_g_test(:,class_number+1) = g_test;
    all_g_train(:,class_number+1) = g_train;
    clearvars -except all_g_test all_g_train class_number;
end

load('digit0.mat');
[~,predict_test] = max(all_g_test');
[~,predict_train] = max(all_g_train');
predict_test = predict_test-1;
predict_train = predict_train-1;

test_error = sum(predict_test'~=test_labels)/num_test*100
train_error = sum(predict_train'~=train_labels)/num_sample*100

%%
clearvars;
class_number = 9;
file = sprintf('digit%d.mat',class_number);
load(file);
for i = 1:3
    figure;
    imshow(reshape(train_imgs(three_most_common_index(i),:),28,28)');
end
clearvars;






