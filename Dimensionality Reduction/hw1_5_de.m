%%
% hw1_5_d
% project the data onto the 15 w* of LDA 
% use projected data of train data to train 6 Gaussian PDF
% predict testing image using simple Gaussian 

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
