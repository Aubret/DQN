package fr.univlyon1.learning;

import fr.univlyon1.agents.AgentDRL;
import fr.univlyon1.memory.ExperienceReplay;
import fr.univlyon1.actorcritic.Learning;
import fr.univlyon1.environment.Interaction;
import fr.univlyon1.memory.prioritizedExperienceReplay.PrioritizedExperienceReplay;
import fr.univlyon1.networks.Approximator;
import fr.univlyon1.networks.EpsilonMultiLayerNetwork;
import fr.univlyon1.networks.Mlp;
import org.agrona.concurrent.Agent;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;

public class TDActorCritic<A> extends TDBatch<A> {

    private Approximator targetActorApproximator ;
    private Approximator criticApproximator ;
    private Approximator targetCriticApproximator ;
    private Approximator cloneCriticApproximator ;

    private double cpt = 0 ;
    private double cumulScoreUp=0;

    private int time = 20;
    private int cpt_time = 0 ;
    private boolean t = true ;

    private Double scoreI ;

    public TDActorCritic(double gamma, Learning<A> learning, ExperienceReplay<A> experienceReplay, int batchSize, int iterations, Approximator criticApproximator,Approximator cloneCriticApproximator) {
        super(gamma, learning, experienceReplay, batchSize, iterations);
        this.targetActorApproximator = this.learning.getApproximator().clone(false);
        this.criticApproximator = criticApproximator ;
        this.approximator = criticApproximator;
        this.targetCriticApproximator = criticApproximator.clone(false);
        this.cloneCriticApproximator = cloneCriticApproximator ;
    }

    /**
     * Extract mini batch
     */
    @Override
    public void learn(){
        int numRows = Math.min(this.experienceReplay.getSize(),this.batchSize);
        int sizeObservation ;
        if(this.lastInteraction != null && this.lastInteraction.getSecondObservation() != null ) {
            INDArray inputAction1 = Nd4j.concat(1, this.lastInteraction.getObservation(), (INDArray) this.learning.getActionSpace().mapActionToNumber(this.lastInteraction.getAction()));
            this.scoreI = this.criticApproximator.getOneResult(inputAction1).getDouble(0) - this.labelize(this.lastInteraction, this.approximator).getDouble(0);
            sizeObservation = this.lastInteraction.getObservation().size(1);
        }else{
            sizeObservation = this.learning.getObservationSpace().getShape()[0];
        }

        if(numRows <1 )
            return ;
        int sizeAction = this.learning.getActionSpace().getSize() ;
        int numColumns =sizeObservation+sizeAction;
        int numColumnsLabels = this.targetCriticApproximator.numOutput();
        //this.learning.getActionSpace().getSize();
        for(int j = 0;j<this.nbrIterations;j++) {
            INDArray observations = Nd4j.zeros(numRows, sizeObservation);
            INDArray inputs = Nd4j.zeros(numRows, numColumns);
            INDArray labels = Nd4j.zeros(numRows, numColumnsLabels);
            for (int i = 0; i < numRows; i++) {
                Interaction<A> interaction = experienceReplay.chooseInteraction();
                INDArray inputAction = Nd4j.concat(1,interaction.getObservation(),(INDArray)this.learning.getActionSpace().mapActionToNumber(interaction.getAction()));
                inputs.putRow(i, inputAction);
                observations.putRow(i, interaction.getObservation());
                labels.putRow(i, this.labelize(interaction, this.approximator));
            }
            //System.out.println("----");
            //INDArray epsilonObsAct = (INDArray)this.criticApproximator.learn(inputs, labels, numRows); // Critic learning
            INDArray scores = (INDArray) this.criticApproximator.learn(inputs, labels, numRows);// Critic learning
            this.experienceReplay.setError(scores);

            if(this.cpt_time%this.time == 0){
                System.out.println("-------------");
                INDArray firstval = ((Mlp) this.criticApproximator).getValues();
                INDArray s1 = firstval.sub(labels);
                Double val1 = s1.muli(s1).meanNumber().doubleValue();
                INDArray newVal = this.criticApproximator.getOneResult(inputs);
                INDArray s2 = newVal.sub(labels);
                Double val2 =s2.muli(s2).meanNumber().doubleValue();
                double meanScore = val1-val2 ;
                cumulScoreUp+=meanScore ;
                System.out.println(meanScore + " -- " +cumulScoreUp);
            }

            this.cloneCriticApproximator.setParams(this.criticApproximator.getParams()); // Dupliquer les paramètres
            INDArray action = this.learning.getApproximator().getOneResult(observations); // L'action du policy network
            INDArray inputAction = Nd4j.concat(1, observations, action);
            INDArray epsilonObsAct = this.cloneCriticApproximator.error(inputAction, Nd4j.create(new double[]{0}), numRows); // erreur
            INDArray epsilonAction = epsilonObsAct.get(NDArrayIndex.all(), NDArrayIndex.interval(sizeObservation, numColumns));
//            System.out.println(epsilonAction.getRow(0));
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
                System.out.println(mean + " -- " + cpt /*+ "---" +((Mlp)this.criticApproximator).getScore()*/);


                //System.out.println(this.criticApproximator);
            }
            this.cpt_time++ ;

        }
    }

    /**
     * Renvoie les labels avec la formue r + lamba*maxQ
     * @param approximator
     */
    @Override
    protected INDArray labelize(Interaction<A> interaction, Approximator approximator){
        //INDArray entry = Nd4j.concat(1,interaction.getObservation(), (INDArray)this.learning.getActionSpace().mapActionToNumber(interaction.getAction())) ;
        // INDArray entryTarget = Nd4j.concat(1,interaction.getObservation(), (INDArray)this.learning.getActionSpace().mapActionToNumber(interaction.getAction())) ;
        INDArray actionTarget = this.targetActorApproximator.getOneResult(interaction.getSecondObservation());
        //INDArray actionTarget = Nd4j.zeros(this.learning.getActionSpace().getSize()).add(0.1);
        INDArray entryCriticTarget = Nd4j.concat(1,interaction.getSecondObservation(), actionTarget) ;
        INDArray res = this.targetCriticApproximator.getOneResult(entryCriticTarget);
        res = res.muli(this.gamma).addi(interaction.getReward()) ;
        return res ;
    }

    public Double getScore(){
        return this.scoreI ;
    }

    public void epoch(){
        /*double alpha = 0.001 ;
        targetActorApproximator.getParams().muli(1.-alpha).addi(this.learning.getApproximator().getParams().mul(alpha));
        targetCriticApproximator.getParams().muli(1.-alpha).addi(this.criticApproximator.getParams().mul(alpha));*/

        /*System.out.println("here");
        System.out.println(targetCriticApproximator.getParams().mul(1-alpha));
        System.out.println(this.criticApproximator.getParams().mul(alpha));
        //System.out.println(p);
        //this.targetCriticApproximator.setParams(p2);
        //this.targetActorApproximator.setParams(p);
        System.out.println(this.targetCriticApproximator.getParams());*/
        this.targetCriticApproximator.setParams(this.criticApproximator.getParams());
        this.targetActorApproximator.setParams(this.learning.getApproximator().getParams());
        //this.cpt_time =0 ;
        //this.targetActorApproximator = this.learning.getApproximator().clone(false);
    }



}
