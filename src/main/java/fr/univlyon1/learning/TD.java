package fr.univlyon1.learning;

import fr.univlyon1.actorcritic.policy.GreedyDiscrete;
import fr.univlyon1.actorcritic.Learning;
import fr.univlyon1.environment.interactions.BetaInteraction;
import fr.univlyon1.environment.interactions.GammaInteraction;
import fr.univlyon1.environment.interactions.Interaction;
import fr.univlyon1.environment.space.SpecificObservation;
import fr.univlyon1.networks.Approximator;
import org.nd4j.linalg.api.ndarray.INDArray;

public class TD<A> implements Algorithm<A> {
    protected Interaction<A> lastInteraction;
    protected double gamma ;
    protected GreedyDiscrete policy;
    protected Learning<A> learning ;
    protected Approximator approximator ;
    protected Interaction<A> previousInteraction ;
    protected Informations informations ;


    public TD(double gamma,Learning<A> learning){
        this.gamma = gamma ;
        this.policy = new GreedyDiscrete();
        this.learning = learning;
        this.approximator = this.learning.getApproximator() ;
        this.informations = new Informations();
    }

    @Override
    public INDArray behave(INDArray input) {
        return this.learning.getApproximator().getOneResult(input);
    }

    @Override
    public void step(INDArray observation, A action,Double time) {
        double dt = 0. ;
        if(this.lastInteraction != null) {
            this.lastInteraction.setSecondAction(action);
            this.lastInteraction.setDt(time-this.lastInteraction.getTime());
            this.informations.setDt(time-this.lastInteraction.getTime());
            this.previousInteraction = this.lastInteraction;
        }
        this.lastInteraction = new GammaInteraction<A>(action,observation,this.learning.getConf().getGamma());
        if(observation instanceof SpecificObservation)
            this.lastInteraction.setIdObserver(((SpecificObservation)observation).getId());
        this.lastInteraction.setTime(time);

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

    public void learn(){
        INDArray res = this.labelize(this.lastInteraction);
        this.learning.getApproximator().learn(this.lastInteraction.getObservation(), res,1);
    }

    @Override
    public Informations getInformation() {
        return this.informations;
    }

    /**
     * Renvoie les labels avec la formue r + lamba*maxQ
     */
    protected INDArray labelize(Interaction<A> interaction){
        INDArray results = approximator.getOneResult(interaction.getSecondObservation());// Trouve l'estimation de QValue
        Integer indice =(int) this.learning.getActionSpace().mapActionToNumber(interaction.getAction());

        INDArray res = this.learning.getApproximator().getOneResult(interaction.getObservation())/*.dup()*/; // résultat précédent dans lequel on change une seule qvalue
        //System.out.println(interaction.getReward() + this.gamma * results.getDouble(this.policy.getAction(results)));
        Double newValue = interaction.getReward() + this.gamma * results.getDouble(this.policy.getAction(results,this.informations));
        //newValue = res.getDouble(indice)+ (new Tanh()).value(newValue - res.getDouble(indice));
        res.putScalar(indice,newValue );
        return res ;
    }

    public Approximator getApproximator(){
        return this.approximator ;
    }

    public Interaction<A> getLastInteraction(){
        return this.lastInteraction;
    }

}
