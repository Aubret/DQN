package fr.univlyon1.learning;


import fr.univlyon1.memory.ExperienceReplay;
import fr.univlyon1.actorcritic.Learning;
import fr.univlyon1.environment.interactions.Interaction;
import fr.univlyon1.networks.Approximator;
import fr.univlyon1.networks.Mlp;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class TDActorCritic<A> extends TDBatch<A> {

    protected Approximator targetActorApproximator ;
    protected Approximator criticApproximator ;
    protected Approximator targetCriticApproximator ;
    protected Approximator cloneCriticApproximator ;

    protected double cpt = 0 ;
    protected double cumulScoreUp=0;

    protected int time = 200;
    protected int cpt_time = 0 ;
    protected boolean t = true ;
    protected boolean epochOne = true;

    protected Double scoreI ;

    public TDActorCritic(double gamma, Learning<A> learning, ExperienceReplay<A> experienceReplay, int batchSize, int iterations, Approximator criticApproximator,Approximator cloneCriticApproximator) {
        super(gamma, learning, experienceReplay, batchSize, iterations);
        this.targetActorApproximator = this.learning.getApproximator().clone(false);
        this.criticApproximator = criticApproximator ;
        this.approximator = criticApproximator;
        this.targetCriticApproximator = criticApproximator.clone(false);
        this.cloneCriticApproximator = cloneCriticApproximator ; // Le clône permet de traiter deux fontions de pertes différentes.
    }


    @Override
    public void learn(){
        this.informations.setModified(true);
        int numRows = Math.min(this.experienceReplay.getSize(),this.batchSize);
        int sizeObservation ;
        if(this.lastInteraction != null && this.lastInteraction.getSecondObservation() != null ) {
            INDArray inputAction1 = Nd4j.concat(1, this.lastInteraction.getObservation(), (INDArray) this.learning.getActionSpace().mapActionToNumber(this.lastInteraction.getAction()));
            this.informations.setScore(this.criticApproximator.getOneResult(inputAction1).getDouble(0) - this.labelize(this.lastInteraction).getDouble(0));
            sizeObservation = this.lastInteraction.getObservation().size(1);
        }else{
            sizeObservation = this.learning.getObservationSpace().getShape()[0];
            //System.out.println(sizeObservation);
        }
        if(numRows < 1 ) {
            return;
        }
        int sizeAction = this.learning.getActionSpace().getSize() ;
        int numColumns =sizeObservation+sizeAction;
        int numColumnsLabels = this.targetCriticApproximator.numOutput();
        //this.learning.getActionSpace().getSize();
        for(int j = 0;j<this.nbrIterations;j++) {
            INDArray observations = Nd4j.zeros(numRows, sizeObservation);
            INDArray inputs = Nd4j.zeros(numRows, numColumns);
            INDArray labels = Nd4j.zeros(numRows, numColumnsLabels);
            for (int i = 0; i < numRows; i++) {
                Interaction<A> interaction = (Interaction<A>)experienceReplay.chooseInteraction();
                INDArray inputAction = Nd4j.concat(1,interaction.getObservation(),(INDArray)this.learning.getActionSpace().mapActionToNumber(interaction.getAction()));
                inputs.putRow(i, inputAction);
                observations.putRow(i, interaction.getObservation());
                labels.putRow(i, /*Nd4j.create(new double[]{interaction.getReward()})*/this.labelize(interaction));
            }
            this.learn_critic(inputs,labels,numRows);
            this.cloneCriticApproximator.setParams(this.criticApproximator.getParams()); // Dupliquer les paramètres
            this.learn_actor(observations,sizeObservation,numColumns,numRows);
            this.cpt_time++ ;
            this.epoch();

        }
    }

    protected void learn_critic(INDArray inputs, INDArray labels, int numRows){
        //System.out.println("----");
        //INDArray epsilonObsAct = (INDArray)this.criticApproximator.learn(inputs, labels, numRows); // Critic learning
        this.criticApproximator.learn(inputs, labels, numRows);// Critic learning
        INDArray scores = this.criticApproximator.getScoreArray();
        this.experienceReplay.setError(scores);

        if(this.cpt_time%this.time == 0){
            System.out.println("-------------");
            INDArray firstval = ((Mlp) this.criticApproximator).getValues().detach();
            INDArray s1 = firstval.sub(labels);
            Double val1 = s1.muli(s1).meanNumber().doubleValue();
            INDArray newVal = this.criticApproximator.getOneResult(inputs);
            INDArray s2 = newVal.sub(labels);
            Double val2 =s2.muli(s2).meanNumber().doubleValue();
            double meanScore = val1-val2 ;
            cumulScoreUp+=meanScore ;
            System.out.println(meanScore + " -- " +cumulScoreUp);
        }
    }

    protected INDArray learn_actor(INDArray observations, int sizeObservation, int numColumns, int numRows){
        INDArray action = this.learning.getApproximator().getOneResult(observations); // L'action du policy network
        this.informations.setEvaluatedActions(action);
        this.informations.setEvaluatedInputs(observations);
        INDArray inputAction = Nd4j.concat(1, observations, action);
        INDArray epsilonObsAct = this.cloneCriticApproximator.error(inputAction, Nd4j.create(new double[]{0}), numRows); // erreur
        INDArray epsilonAction = epsilonObsAct.get(NDArrayIndex.all(), NDArrayIndex.interval(sizeObservation, numColumns));
        INDArray old =null;
        if(this.cpt_time%this.time == 0) {
            old = this.cloneCriticApproximator.getOneResult(inputAction);
        }

        this.learning.getApproximator().learn(observations, epsilonAction, numRows); //Policy learning

        if(this.cpt_time%this.time == 0 ) {
            action = this.learning.getApproximator().getOneResult(observations); // L'action du policy network
            inputAction = Nd4j.concat(1, observations, action);
            INDArray intermediaire = this.cloneCriticApproximator.getOneResult(inputAction).subi(old);//must be positive
            Number mean = intermediaire.meanNumber();
            cpt += mean.doubleValue();
            System.out.println(mean + " -- " + cpt);
        }
        return epsilonObsAct ;
    }

    /**
     * Renvoie les labels avec la formue r + lamba*maxQ
     */
    @Override
    protected INDArray labelize(Interaction<A> interaction){
        /*if(this.epochOne){
            return Nd4j.create(new double[]{interaction.getReward()});
        }*/
        //INDArray entry = Nd4j.concat(1,interaction.getObservation(), (INDArray)this.learning.getActionSpace().mapActionToNumber(interaction.getAction())) ;
        // INDArray entryTarget = Nd4j.concat(1,interaction.getObservation(), (INDArray)this.learning.getActionSpace().mapActionToNumber(interaction.getAction())) ;
        INDArray actionTarget = this.targetActorApproximator.getOneResult(interaction.getSecondObservation());
        //INDArray actionTarget = Nd4j.zeros(this.learning.getActionSpace().getSize()).add(0.1);
        INDArray entryCriticTarget = Nd4j.concat(1,interaction.getSecondObservation(), actionTarget) ;
        INDArray res = this.targetCriticApproximator.getOneResult(entryCriticTarget);
        /*System.out.println("---");
        System.out.println(res);
        System.out.println(interaction.getReward());
        System.out.println(res.muli(this.gamma).getDouble(0));*/
        /*INDArray res1 = res.dup().muli(this.gamma).addi(Nd4j.create(new double[]{interaction.getReward()}) );
        System.out.println(res.addi(interaction.getReward()).getDouble(0));
        System.out.println(res1.getDouble(0)); */
        res.muli(this.gamma).addi(interaction.getReward());
        return res ;
    }

    public Double getScore(){
        return this.scoreI ;
    }

    public void epoch(){
        double alpha = 0.01 ;
        targetActorApproximator.getParams().muli(1.-alpha).addi(this.learning.getApproximator().getParams().mul(alpha));
        targetCriticApproximator.getParams().muli(1.-alpha).addi(this.criticApproximator.getParams().mul(alpha));
        this.epochOne = false;
    }


    public Approximator getTargetActorApproximator() {
        return targetActorApproximator;
    }

    public void setTargetActorApproximator(Approximator targetActorApproximator) {
        this.targetActorApproximator = targetActorApproximator;
    }

    public Approximator getTargetCriticApproximator() {
        return targetCriticApproximator;
    }

    public void setTargetCriticApproximator(Approximator targetCriticApproximator) {
        this.targetCriticApproximator = targetCriticApproximator;
    }
}
