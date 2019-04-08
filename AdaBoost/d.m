load('Adaboost250.mat');
a10 = 128*ones(c,d);
figure;
for k = 1:c
    for wl = 2:T+1
        jBest = Alf(k,1,wl);
        uBest = Alf(k,2,wl);
        if uBest <= 51 %regular weak learner
            a10(k,jBest) = 255;
        else %twin weak learner
            a10(k,jBest) = 0;
        end
    end
    ak = reshape(a10(k,:),28,28)';
    subplot(2,5,k);
    imshow(uint8(ak));
    title(['Array $$\textbf{a}$$ for the Binary Classifier ' num2str(k)],'interpreter','latex');
end
