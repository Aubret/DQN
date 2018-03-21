package fr.univlyon1.actorcritic.policy;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Random;

public class NoisyGreedy implements Policy {
    private double std ;
    private double mean;
    private Random random ;

    public NoisyGreedy(double std, double mean ,long seed){
        this.random = new Random(seed);
        this.std = std ;
        this.mean = mean ;
    }

    @Override
    public INDArray getAction(INDArray results) {
        for(int i = 0 ; i < results.size(1) ; i++){
            results.putScalar(i,results.getDouble(i)+random.nextGaussian()*this.std+this.mean);
        }
        return results ;
    }
}
