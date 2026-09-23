
function output_signal = mp3_codec( input_signal, Fs )

    %% Pseudo QMF filters bank
    hn = PQMF32_prototype;
    PQMF32_Gfilters = zeros(32, 512);
    for i = 0:31
        t2 = ((2*i+1)*pi/(2*32))*((0:511)+16);
        PQMF32_Gfilters(i+1,:) = hn.*cos(t2);
    end

    n_frames = floor((length(input_signal)-512+32)/32);
    subbands=zeros(n_frames,32);
    for i=1:n_frames
         if rem(i, 100) == 0
            fprintf('Analyzing frame %5d / %5d\n', i,n_frames)
         end
        input_frame = input_signal((i-1)*32+1:(i-1)*32+512);
        subbands(i,:) = (PQMF32_Gfilters*input_frame)';
    end

    n_frames = fix(n_frames/12)*12;
    quantized_subbands=zeros(n_frames,32);
    for k=1:12:n_frames
         if rem(k-1, 120) == 0
            fprintf('Processing frame %5d %5d\n', k,n_frames)
         end
        [scale_factors,tmp] = max(abs(subbands(k:k+11,:)));
        frame = input_signal(176+(k-1)*32:176+(k-1)*32+511);
        SMR = MPEG1_psycho_acoustic_model1(frame);
     
        % Allocating bits for a target bit rate of 192 kbits/s
        N_bits = MPEG1_bit_allocation(SMR, 192000);

        % Adaptive perceptual uniform quantization, using a mid-thread
        % quantizer in [-Max,+Max]
        for j=1:32 % for each sub-band
            if N_bits(j)~=0
                codes= uencode(subbands(k:k+11,j),N_bits(j),...
                    scale_factors(j),'signed');
                quantized_subbands(k:k+11,j) = ...
                    udecode(codes,N_bits(j),scale_factors(j));
            else
                quantized_subbands(k:k+11,j) = 0;
            end
        end
    end

    output_signal = zeros(size(input_signal));
    for i=1:n_frames
        output_frame = PQMF32_Gfilters'*quantized_subbands(i,:)';
        output_signal((i-1)*32+1:(i-1)*32+512)= ...
            output_signal((i-1)*32+1:(i-1)*32+512)+output_frame;
    end


