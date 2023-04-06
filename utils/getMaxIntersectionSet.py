import os

wavelength = [
449.466,
451.022,
452.598,
454.194,
455.809,
457.443,
459.095,
460.765,
462.452,
464.157,
465.879,
467.618,
469.374,
471.146,
472.935,
474.742,
476.564,
478.404,
480.261,
482.136,
484.027,
485.937,
487.865,
489.811,
491.776,
493.76,
495.763,
497.787,
499.832,
501.898,
503.985,
506.095,
508.228,
510.384,
512.565,
514.771,
517.003,
519.261,
521.547,
523.86,
526.203,
528.576,
530.979,
533.414,
535.882,
538.383,
540.919,
543.49,
546.098,
548.743,
551.426,
554.149,
556.912,
559.718,
562.565,
565.457,
568.393,
571.375,
574.405,
577.482,
580.609,
583.786,
587.015,
590.296,
593.631,
597.021,
600.467,
603.97,
607.532,
611.153,
614.835,
618.578,
622.385,
626.255,
630.19,
634.192,
638.261,
642.399,
646.606,
650.883,
655.232,
659.654,
664.15,
668.72,
673.367,
678.09,
682.891,
687.771,
692.73,
697.77,
702.892,
708.096,
713.384,
718.755,
724.212,
729.755,
735.384,
741.101,
746.906,
752.8,
758.784,
764.857,
771.022,
777.278,
783.626,
790.067,
796.6,
803.228,
809.949,
816.764,
823.675,
830.68,
837.781,
844.978,
852.271,
859.659,
867.144,
874.725,
882.403,
890.176,
898.047,
906.013,
914.076,
922.235,
930.49,
938.84,
947.285,
955.825
]

embedingSelectedLog = '../output_sHaddWater.log'
handselect = sorted([1, 13, 25, 52, 76, 92, 99, 105, 109])
minInterval = 3
threshold = 4
with open(embedingSelectedLog, 'r')as f:
    lines = f.readlines()
    for line in lines:
        if line.find('best bands:') != -1:
            pos = line.find('best bands:')
            selectRes = line[pos + 13:].split(" ")
            # print(selectRes)
            selectBnads = []
            for i in selectRes:
                i = i.strip()
                if i == '':
                    continue
                selectBnads.append(int(i))
                if len(selectBnads) == 9:
                    break
            interSet = []
            tmphandselect = handselect.copy()
            for y in selectBnads:
                for x in tmphandselect:
                    if abs(x - y) <= minInterval:
                        interSet.append(y)
                        tmphandselect.remove(x)
            if len(interSet) >= threshold:
                pos = line.find('epoch:')
                epoch = line[pos + 7:].split(" ")[0]
                print(epoch, selectBnads)
                wave = [wavelength[i] for i in tmphandselect]
                print(wave)