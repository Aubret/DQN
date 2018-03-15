package main.java.fr.univlyon1.actorcritic.policy;

import main.java.fr.univlyon1.environment.ActionSpace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Random;

public class Egreedy implements Policy{
    private double epsilon;
    private Random random ;

    public Egreedy(double epsilon,long seed){
        this.epsilon = epsilon ;
        this.random = new Random(seed);
    }
    @Override
    public Integer getAction(INDArray results) {
        int indice ;
        if(this.random.nextDouble() < this.epsilon) {
            indice = random.nextInt(results.size(1));
        }else
            indice = Nd4j.argMax(results).getInt(0);
        return indice ;
    }
}
