%%
clc;
clear all;
% load image 
% train: 6 folders each 40 images
% test: 6 folders each 10 images
trainData = zeros(240,2500);
testData = zeros(60,2500);

for i = 0:5
    trainData(i*40+1:(i+1)*40,:) = load_trainset(i);
    testData(i*10+1:(i+1)*10,:) = load_testset(i);
end

clearvars -except trainData testData


%%
% hw1_5_a
% draw 16 eigenfaces of PCA-> the 16 eigenvectors of the covariance matrix 
% of the whole training data 
% training data dimesion 1*2500 -> covariance matrix 2500*2500
% -> eigenvector of covariance matrix 1*2500
% uCovTr each column is the eigenvector, max to min from left to right
% each face is 1*2500 eigenvector -> 50*50 image
cov_trainData = cov(trainData);
[uCovTr,sCovTr,vCovTr] = svd(cov_trainData);
first_nth_eig = 16;
% 16 eigenFaces: plot of the first 16 largest eigenvectors 
for i = 1:first_nth_eig
    subplot(4,4,i);
    imagesc(reshape(uCovTr(:,i),50,50));
    colormap('gray');
end
%%
% hw1_5_b
% draw the 15 eigenfaces of LDA, 15 by 6 classes take 2 
% non invertible case of LDA -> RDA
% ex: class1 vs class0
% SB = (u1 - u0)(u1 - u0).T, 2500*2500
% SW = cov1 + cov0, 2500*2500
% SW' = SW + gamma*I
% w* = eigenvector of the largest eigenvalue of (SW')^-1 * SB, 1*2500
% -> 15 w* for 6 classes

num_classes = 6;
mean_classes = zeros(6,2500);
% each covariance matrix of each class 2500*2500
cov_classes = zeros(2500,2500*6);
for i = 1:num_classes
    mean_classes(i,:) = mean(trainData((i-1)*40+1:i*40,:));
    cov_classes(:,(i-1)*2500+1:i*2500) = cov(trainData((i-1)*40+1:i*40,:));
end

% non invertible case RDA 
gamma = 1;

plot_count = 1;
all_ij_max_u_RDA = zeros(2500,15);

for i = 1:num_classes-1
    mean_ith_class = mean_classes(i,:);
    cov_ith_class =  cov_classes(:,(i-1)*2500+1:i*2500);
    for j = i+1:num_classes 
        mean_jth_class = mean_classes(j,:);
        cov_jth_class = cov_classes(:,(j-1)*2500+1:j*2500);
        SB_ij = (mean_jth_class-mean_ith_class)'*...
            (mean_jth_class - mean_ith_class);
        SW_ij =  cov_ith_class + cov_jth_class + gamma*eye(2500);
        [u_RDA,s_RDA,v_RDA] = svd(SW_ij\SB_ij);
        all_ij_max_u_RDA(:,plot_count) = u_RDA(:,1);
        subplot(4,4,plot_count);
        imagesc(reshape(u_RDA(:,1),50,50));
        colormap('gray');
        plot_count = plot_count + 1;
    end
end
subplot(4,4,16);
imagesc(zeros(50,50));
colormap('gray');


%%
% hw1_5_c
% use PCA choosing k = 15 (15 eigenvectors) for dimension reduction
% use projected data to train a simple Gaussian classifier
% i.e 6 Gaussian PDF 
% input a test image vector -> project using training set PCA 
% -> classify to one of the 6 classes by choosing the largest PDF
% result (highest probability) of each of the 6 Gaussian PDF of 
% the training set
first_15th_eig = uCovTr(:,1:15);
% data - mean before projection 
pca_trainData = (trainData-mean(trainData)) * first_15th_eig;
assert(all(size(pca_trainData) == [240,15]));

pca_mean_classes = zeros(6,15);
pca_cov_classes = zeros(15,15*6);

% calculate the 6 mean, covariance matrix of the 6 projected classes 
for i = 1:6
    trainData_iClass = pca_trainData((i-1)*40+1:i*40,:);
    pca_mean_classes(i,:) = mean(trainData_iClass);
    pca_cov_classes(:,(i-1)*15+1:i*15) =  cov(trainData_iClass);
end

pca_testData = (testData - mean(trainData)) * first_15th_eig;
predict_testData = zeros(size(pca_testData,1),1);
predictClasses = zeros(1,6);

all_classes_pdf = zeros(size(testData,1), 6);

for i = 1:size(pca_testData,1)
    xtest = pca_testData(i,:);
    classes_pdf = zeros(1,6);
    for j = 1:6
        classes_pdf(j) = mvnpdf(xtest,pca_mean_classes(j,:),pca_cov_classes(:,(j-1)*15+1:j*15));
        all_classes_pdf(i,j) = classes_pdf(j);
    end
    [maxPdf, maxIndex] = max(classes_pdf);
    predict_testData(i) = maxIndex;
    predictClasses(maxIndex) = predictClasses(maxIndex) + 1;
end

% class_error: each item is the number of misclassified images for each
% class 
class_error = zeros(1,6);
for i = 1:6
    for j = 1:10
        if predict_testData((i-1)*10 + j) ~= i
            class_error(i) = class_error(i) + 1;
        end
    end
end
class_error_percent = class_error / 10 * 100;
disp(sum(class_error) / 60 * 100);

%%
% hw1_5_d
% project the data onto the 15 w* of LDA 
% use projected data of train data to train 6 Gaussian PDF
% predict testing image using simple Gaussian 
% pca allows data to less easy to be non invertible, -> no need for RDA 
% (add gamma*I) use LDA, 


lda_trainData = (trainData-mean(trainData)) * all_ij_max_u_RDA;
assert(all(size(lda_trainData) == [240,15]));
lda_mean_classes = zeros(6,15);
lda_cov_classes = zeros(15,15*6);

for i = 1:6
    lda_trainData_iClass = lda_trainData((i-1)*40+1:i*40,:);
    lda_mean_classes(i,:) = mean(lda_trainData_iClass);
    lda_cov_classes(:,(i-1)*15+1:i*15) =  cov(lda_trainData_iClass);
end

lda_testData = (testData - mean(trainData)) * all_ij_max_u_RDA;
lda_predict_testData = zeros(size(lda_testData,1),1);
lda_predictClasses = zeros(1,6);

lda_all_classes_pdf = zeros(size(testData,1), 6);

for i = 1:size(lda_testData,1)
    xtest = lda_testData(i,:);
    classes_pdf = zeros(1,6);
    for j = 1:6
        classes_pdf(j) = mvnpdf(xtest,lda_mean_classes(j,:),lda_cov_classes(:,(j-1)*15+1:j*15));
        lda_all_classes_pdf(i,j) = classes_pdf(j);
    end
    [maxPdf, maxIndex] = max(classes_pdf);
    lda_predict_testData(i) = maxIndex;
    lda_predictClasses(maxIndex) = lda_predictClasses(maxIndex) + 1;
end

lda_class_error = zeros(1,6);
for i = 1:6
    for j = 1:10
        if lda_predict_testData((i-1)*10 + j) ~= i
            lda_class_error(i) = lda_class_error(i) + 1;
        end
    end
end

lda_class_error_percent = lda_class_error / 10 * 100;
disp(sum(lda_class_error) / 60 * 100);

%%
% hw1_5_e
% first dimension reduction using PCA, choosing k = 30, 
% the first 30 eigenvector, then use LDA dimension reduction data to 
% train a Gaussian PDF, use the 6 PDF to predict the testing images
% to class 


clearvars -except trainData testData uCovTr num_classes

% pca training data to 30 dimension
first_30th_eig = uCovTr(:,1:30);
pca_trainData = (trainData-mean(trainData)) * first_30th_eig;
assert(all(size(pca_trainData) == [240,30]));

% find the 15 w vectors of lda of the pca training data 
mean_classes = zeros(6,30);
cov_classes = zeros(30,30*6);

for i = 1:num_classes
    trainData_iClass = pca_trainData((i-1)*40+1:i*40,:);
    mean_classes(i,:) = mean(trainData_iClass);
    cov_classes(:,(i-1)*30+1:i*30) =  cov(trainData_iClass);
end

ij_count = 1;
all_ij_w_star = zeros(30,15);

for i = 1:num_classes-1
    mean_ith_class = mean_classes(i,:);
    cov_ith_class =  cov_classes(:,(i-1)*30+1:i*30);
    assert(all(size(mean_ith_class)==[1,30]));
    assert(all(size(cov_ith_class)==[30,30]));
    for j = i+1:num_classes 
        mean_jth_class = mean_classes(j,:);
        cov_jth_class = cov_classes(:,(j-1)*30+1:j*30);
        assert(all(size(mean_jth_class)==[1,30]));
        assert(all(size(cov_jth_class)==[30,30]));
        all_ij_w_star(:,ij_count) = (cov_jth_class + cov_ith_class)\...
            (mean_jth_class - mean_ith_class)';
        ij_count = ij_count + 1;
    end
end

% lda the pca training data
lda_pca_trainData = (pca_trainData-mean(pca_trainData)) * all_ij_w_star;
assert(all(size(lda_pca_trainData) == [240,15]));

lda_pca_testData = ((testData - mean(trainData))*first_30th_eig ...
    - mean(pca_trainData))*all_ij_w_star;
assert(all(size(lda_pca_testData)==[60,15]));

% find the 6 classes mean covariance of the
lda_mean_classes = zeros(6,15);
lda_cov_classes = zeros(15,15*6);

for i = 1:num_classes
    trainData_iClass = lda_pca_trainData((i-1)*40+1:i*40,:);
    lda_mean_classes(i,:) = mean(trainData_iClass);
    lda_cov_classes(:,(i-1)*15+1:i*15) =  cov(trainData_iClass);
end

all_classes_pdf = zeros(60,6);
predictClasses = zeros(1,6);
for i = 1:size(lda_pca_testData,1)
    xtest = lda_pca_testData(i,:);
    classes_pdf = zeros(1,6);
    for j = 1:6
        classes_pdf(j) = mvnpdf(xtest,lda_mean_classes(j,:),lda_cov_classes(:,(j-1)*15+1:j*15));
        all_classes_pdf(i,j) = classes_pdf(j);
    end
    [maxPdf, maxIndex] = max(classes_pdf);
    predict_testData(i) = maxIndex;
    predictClasses(maxIndex) = predictClasses(maxIndex) + 1;
end

class_error = zeros(1,6);
for i = 1:6
    for j = 1:10
        if predict_testData((i-1)*10 + j) ~= i
            class_error(i) = class_error(i) + 1;
        end
    end
end
class_error_percent = class_error / 10 * 100;
disp(sum(class_error) / 60 * 100);



