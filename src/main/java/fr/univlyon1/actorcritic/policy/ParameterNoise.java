package fr.univlyon1.actorcritic.policy;

import fr.univlyon1.actorcritic.policy.Policy;
import fr.univlyon1.environment.space.ActionSpace;
import fr.univlyon1.learning.Informations;
import fr.univlyon1.networks.Approximator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.GaussianDistribution;
import org.nd4j.linalg.api.rng.distribution.impl.NormalDistribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;


/**
 * From paper "Parameter space Noise for exploration"
 * @param <A>
 */
public class ParameterNoise<A> implements Policy<A> {

    // Must have weights
    protected Approximator greedyPolicy ;
    protected Approximator modifiedPolicy ;
    protected long seed ;
    protected int schedule ;
    protected int timer ;
    protected double variance;
    protected double alpha ;
    protected double thresholdDistance ;

    public ParameterNoise(double std, long seed, ActionSpace<A> actionSpace, Approximator greedyPolicy, int schedule){
        this.greedyPolicy = greedyPolicy ;
        this.modifiedPolicy = greedyPolicy.clone();
        this.seed = seed ;
        this.schedule = schedule ;
        this.timer = 0 ;
        this.variance = 0.01 ;
        this.alpha = 1.01 ;
        thresholdDistance = 0.2;
    }

    @Override
    public Object getAction(INDArray results, Informations informations) {
        this.timer+=informations.getDt() ;
        if(this.timer > this.schedule){
            this.timer = 0 ;
            INDArray weights = this.greedyPolicy.getParams().dup() ;
            INDArray distrib = (new NormalDistribution(0., variance)).sample(weights.shape());
            weights.addi(distrib);
            this.modifiedPolicy.setParams(weights);
            this.changeVariance(informations);
        }
        INDArray res =(INDArray) this.modifiedPolicy.getAction(results,informations);
        return res;
    }

    protected void changeVariance(Informations informations){
        if(informations.getEvaluatedInputs() == null || !informations.isModified())
            return ;
        informations.setModified(false);

        INDArray actions = (INDArray)this.modifiedPolicy.getAction(informations.getEvaluatedInputs(),informations);
        int N = actions.columns();
        int statesN = actions.rows();
        INDArray diff = actions.sub(informations.getEvaluatedActions());
        INDArray expect =  (Transforms.pow(diff, 2)).sum(0).divi(statesN);
        INDArray dist = Transforms.sqrt(expect.sum(1).divi(N));
        //System.out.println(dist.getDouble(0)+"  vs "+this.thresholdDistance);
        if(dist.getDouble(0) > this.thresholdDistance){
            this.variance = (1./this.alpha )* this.variance ;
        }else{
            this.variance = this.alpha*this.variance;
        }
    }
}
