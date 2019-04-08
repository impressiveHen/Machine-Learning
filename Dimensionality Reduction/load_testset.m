function testset = load_testset(num)
    % num: (int) folder number
    % trainset: matrix 10*2500, num_images*vectorized dimension
    % 50*50 matrix image -> 1*2500 vector image
    testset = zeros(10, 50*50);
    for i = 1:10
        img = im2double(imread(strcat('D:\ucsdCourse\2019WinterQuarter\ece271bSL2\hw\testset\subset',int2str(num+6),'\person_',int2str(num+1),'_',int2str(i),'.jpg')));
        testset(i,:) = reshape(img, 1, []);
    end
end