    function op=LOE_FN(ip_path, e_path)
        %input image
        ipic=imread(ip_path);
        ipic=double(ipic);
        %enhanced image
        epic=imread(e_path);
        epic=double(epic);
        
        [m,n,k]=size(ipic);

        
        %get the local maximum for each pixel of the input image
        win=7;
        imax=round(max(max(ipic(:,:,1),ipic(:,:,2)),ipic(:,:,3)));
        imax=getlocalmax(imax,win);
        %get the local maximum for each pixel of the enhanced image
        emax=round(max(max(epic(:,:,1),epic(:,:,2)),epic(:,:,3)));
        emax=getlocalmax(emax,win);
        
        %get the downsampled image
        blkwin=50;
        mind=min(m,n);
        step=floor(mind/blkwin);% the step to down sample the image
        blkm=floor(m/step);
        blkn=floor(n/step);
        ipic_ds=zeros(blkm,blkn);% downsampled of the input image
        epic_ds=zeros(blkm,blkn);% downsampled of the enhanced image
        LOE=zeros(blkm,blkn);%
        
        for i=1:blkm
            for j=1:blkn
                ipic_ds(i,j)=imax(i*step,j*step);
                epic_ds(i,j)=emax(i*step,j*step);
            end
        end

        for i=1:blkm
            for j=1:blkn%bug
                flag1=ipic_ds>=ipic_ds(i,j);
                flag2=epic_ds>=epic_ds(i,j);
                flag=(flag1~=flag2);
                LOE(i,j)=sum(flag(:));
            end
        end
        
        LOE=mean(LOE(:))
        op=LOE;
    end
    
    function output=getlocalmax(pic,win)
        [m,n]=size(pic);
        extpic=getextpic(pic,win);
        output=zeros(m,n);
        for i=1+win:m+win
            for j=1+win:n+win
                modual=extpic(i-win:i+win,j-win:j+win);
                output(i-win,j-win)=max(modual(:));
            end
        end
    end
    
    function output=getextpic(im,win_size)
        [h,w,c]=size(im);
        extpic=zeros(h+2*win_size,w+2*win_size,c);
        extpic(win_size+1:win_size+h,win_size+1:win_size+w,:)=im;
        for i=1:win_size%extense row
            extpic(win_size+1-i,win_size+1:win_size+w,:)=extpic(win_size+1+i,win_size+1:win_size+w,:);%top edge
            extpic(h+win_size+i,win_size+1:win_size+w,:)=extpic(h+win_size-i,win_size+1:win_size+w,:);%botom edge
        end
        for i=1:win_size%extense column
            extpic(:,win_size+1-i,:)=extpic(:,win_size+1+i,:);%left edge
            extpic(:,win_size+w+i,:)=extpic(:,win_size+w-i,:);%right edge
        end
        output=extpic;
    end
    


low_light_folder= '/MATLAB Drive/NLIEE/input_images/'

low_light_files = dir(fullfile(low_light_folder, '*.png'));  % Adjust the pattern if needed
all_items = dir(fullfile('/MATLAB Drive/NLIEE/', 'op_*'));

% Filter only directories that do not end with ".zip"
enhanced_folders = all_items([all_items.isdir] & ~endsWith({all_items.name}, '.zip'));


Lambda=[];
Gamma=[];
Sigma=[];
filename={};
Quality_Scores=[];
idx=1;
for j=1: length(enhanced_folders)
    e_folder= enhanced_folders(j);

    for i = 1:length(low_light_files)
        % Get the file name for low-light image
        low_light_file = low_light_files(i).name;
        low_light_path = fullfile(low_light_folder, low_light_file);
        enhanced_path = fullfile(e_folder.folder, e_folder.name, low_light_file);
        img1 = imread(enhanced_path);
        img2 = imread(low_light_path);
        % quality_score = NLIEE_SVR(img1,img2)
        quality_score= LOE_FN(low_light_path, enhanced_path)

        filename{idx}=low_light_file;
        Quality_Scores{idx}= quality_score;

        input_str = e_folder.name;

        % Extract values after G, L, and S
        G_value = regexp(input_str, 'G-(\d+)', 'tokens', 'once');
        L_value = regexp(input_str, 'L-(\d+)', 'tokens', 'once');
        S_value = regexp(input_str, 'S-(\d+)', 'tokens', 'once');
        
        % Convert from cell array to numeric values
        Gamma(idx) = str2double(G_value{1});
        Lambda(idx) = str2double(L_value{1});
        Sigma(idx) = str2double(S_value{1});
        idx=idx+1;
    end
end




dataTable = table(Lambda', Gamma', Sigma', filename', Quality_Scores', ...
                  'VariableNames', {'Lambda', 'Gamma', 'Sigma', 'Filename', 'Quality_Scores'});

% Specify the CSV file name
csvFileName = '/MATLAB Drive/NLIEE/LOE_output.csv';

% Write the table to a CSV file
writetable(dataTable, csvFileName);

disp(['CSV file created: ', csvFileName]);