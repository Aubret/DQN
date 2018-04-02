package fr.univlyon1.actorcritic.policy;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Random;

public class NoisyGreedy implements Policy {
    protected double std ;
    protected double mean;
    protected Random random ;
    protected Policy greedy ;

    public NoisyGreedy(double std, double mean ,long seed,Policy greedy){
        this.random = new Random(seed);
        this.std = std ;
        this.mean = mean ;
        this.greedy = greedy ;
    }

    @Override
    public INDArray getAction(INDArray inputs) {
        INDArray results = (INDArray)this.greedy.getAction(inputs);
        for(int i = 0 ; i < results.size(1) ; i++){
            double newVal = results.getDouble(i)+random.nextGaussian()*this.std+this.mean ;
            newVal = Math.min(1.,Math.max(-1,newVal)) ; // Sinon on sort de 1 ett -1 et ca fai tbuguer
            results.putScalar(i,newVal);
        }
        return results ;
    }
}
