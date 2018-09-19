package fr.univlyon1.memory.prioritizedExperienceReplay;

import fr.univlyon1.environment.interactions.Interaction;
import fr.univlyon1.memory.SequentialExperienceReplay;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;

public class SequentialPrioritizedExperienceReplay<A> extends SequentialExperienceReplay<A>{
    protected StochasticPrioritizedExperienceReplay<A> prioritized ;
    protected int num ;
    protected int learn ;

    public SequentialPrioritizedExperienceReplay(int maxSize, ArrayList<String> file, int sequenceSize, int backpropSize, long seed,int learn) {
        super(maxSize, file, sequenceSize, backpropSize, seed);
        this.prioritized = new StochasticPrioritizedExperienceReplay<A>(maxSize/5,seed,file);
        //this.prioritized = new StochasticForPrioritized<A>(20000,seed,file,this);
        this.learn = learn ;
        this.num = 0;
    }

    @Override
    public void addInteraction(Interaction<A> interaction) {
        if (this.interactions.size() == this.maxSize) {
            Interaction<A> remove = this.interactions.get(0);
            this.prioritized.removeInteraction(remove);
            this.interactions.remove(remove);
        }
        if(this.num == 3) {
            if(! (this.prioritized instanceof StochasticForPrioritized)) {
                this.prioritized.addInteractionNotTaken(interaction);
            }else {
                this.prioritized.addInteraction(interaction);
            }
            this.num = 0 ;
        }
        this.interactions.add(interaction);
        this.num++ ;
    }

    protected Interaction<A> choose(){
        Interaction<A> start= this.prioritized.chooseInteraction();
        this.cursor = this.interactions.indexOf(start);
        return start;
    }

    public void setError(INDArray errors, ArrayList<Integer> backpropNumber, int backward,ArrayList<ArrayList<Interaction<A>>> total) {
        int cursor = 0;
        INDArray errorsNew = Nd4j.zeros(total.size());
        for(int i=0; i< total.size();i++){
            INDArray mean = Nd4j.zeros(1);
            for( int j = 0; j < backpropNumber.get(i)-1;j++){ //-1 pcq le premier S est pas envoyé au critic
                //mean.addi(errors.getDouble(cursor));
                cursor++;
            }
            /*mean.divi(backpropNumber.get(i)-1);
            errorsNew.put(i,mean);*/
            errorsNew.putScalar(i,errors.getDouble(cursor-1));
        }
        //System.out.println(errorsNew);
        this.prioritized.setError(errorsNew);
    }

    public boolean initChoose(){ // Toujours appeler avant les chooseInteraction
        if(this.prioritized.getSize() == 0)
            return false ;
        if(this.interactions.size() <= minForward)
            return false ;
        if(this.interactions.get(this.interactions.size()-1).getTime() - this.interactions.get(0).getTime() < this.sequenceSize )
            return false ;

        // On vérifie qu ele curseur actuel suffit à proposer une séquence complète
        Interaction<A> start = this.choose();
        Double dt = this.interactions.get(this.interactions.size()-1).getTime() - start.getTime() ;
        int cpt = 0 ;
        while(dt < this.sequenceSize || (this.interactions.size() - cursor <= minForward) || this.interactions.get(cursor+1).getTime()-start.getTime() > this.sequenceSize){
            if(cpt == 10){
                this.prioritized.repushLast();// On replace le dernier choisi
                cursor=0 ; // On veut limiter le nombre de recherches aléatoires
                break;
            }else {
                this.prioritized.repushLast();// On replace le dernier choisi
                start = this.choose();
                dt = this.interactions.get(this.interactions.size()-1).getTime() - start.getTime() ;
                //cursor = 0; // On a déjà vérifié que c'était possible avec 0
                cpt++ ;
            }
        }
        start = this.interactions.get(cursor);
        this.startTime = start.getTime() ;
        this.backpropNumber = 0 ;
        this.forwardNumber = 0 ;
        this.tmp = new ArrayList<>();
        return true ;
    }

    public boolean isAvailable(Interaction<A> interaction){
        int curs = this.interactions.indexOf(interaction);
        Double dt = this.interactions.get(this.interactions.size()-1).getTime() - interaction.getTime() ;
        if(dt < this.sequenceSize || (this.interactions.size() - curs <= minForward) || this.interactions.get(curs+1).getTime()-interaction.getTime() > this.sequenceSize)
            return false ;
        return true ;
    }


}
