package fr.univlyon1.learning;

import fr.univlyon1.actorcritic.policy.GreedyDiscrete;
import fr.univlyon1.actorcritic.Learning;
import fr.univlyon1.environment.Interaction;
import fr.univlyon1.networks.Approximator;
import org.nd4j.linalg.api.ndarray.INDArray;

public class TD<A> implements Algorithm<A> {
    protected Interaction<A> lastInteraction;
    protected double gamma ;
    protected GreedyDiscrete policy;
    protected Learning<A> learning ;
    protected Approximator approximator ;

    public TD(double gamma,Learning<A> learning){
        this.gamma = gamma ;
        this.policy = new GreedyDiscrete();
        this.learning = learning;
        this.approximator = this.learning.getApproximator() ;
    }

    @Override
    public void step(INDArray observation, A action) {
        //this.lastInteraction.setSecondAction(action);
        this.lastInteraction = new Interaction<A>(action,observation);
    }

    @Override
    public void evaluate(INDArray input, Double reward) {
        if(this.lastInteraction != null) { // Avoir des interactions complètes
            this.lastInteraction.setSecondObservation(input);
            this.lastInteraction.setReward(reward);
        }
    }

    @Override
    public void epoch() {

    }

    @Override
    public Double getScore() {
        return null;
    }

    public void learn(){
        INDArray res = this.labelize(this.lastInteraction,this.approximator);
        this.learning.getApproximator().learn(this.lastInteraction.getObservation(), res,1);
    }

    /**
     * Renvoie les labels avec la formue r + lamba*maxQ
     * @param approximator
     */
    protected INDArray labelize(Interaction<A> interaction,Approximator approximator){
        INDArray results = approximator.getOneResult(interaction.getSecondObservation());// Trouve l'estimation de QValue
        Integer indice =(int) this.learning.getActionSpace().mapActionToNumber(interaction.getAction());

        INDArray res = this.learning.getApproximator().getOneResult(interaction.getObservation())/*.dup()*/; // résultat précédent dans lequel on change une seule qvalue
        //System.out.println(interaction.getReward() + this.gamma * results.getDouble(this.policy.getAction(results)));
        Double newValue = interaction.getReward() + this.gamma * results.getDouble(this.policy.getAction(results));
        //newValue = res.getDouble(indice)+ (new Tanh()).value(newValue - res.getDouble(indice));
        res.putScalar(indice,newValue );
        return res ;
    }

    public Approximator getApproximator(){
        return this.approximator ;
    }

}
