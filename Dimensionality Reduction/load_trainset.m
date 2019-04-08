function trainset = load_trainset(num)
    % num: (int) folder number
    % trainset: matrix 40*2500, num_images*vectorized dimension
    % 50*50 matrix image -> 1*2500 vector image
    trainset = zeros(40, 50*50);
    for i = 1:40
        img = im2double(imread(strcat('D:\ucsdCourse\2019WinterQuarter\ece271bSL2\hw\trainset\subset',int2str(num),'\person_',int2str(num+1),'_',int2str(i),'.jpg')));
        trainset(i,:) = reshape(img, 1, []);
    end
end

