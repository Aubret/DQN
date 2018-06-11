package fr.univlyon1.memory;

import fr.univlyon1.environment.Interaction;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.Random;


public class SequentialExperienceReplay<A> extends ExperienceReplay<A>{
    protected ArrayList<Interaction<A>> interactions ;
    protected ArrayList<Interaction<A>> tmp ;
    protected Integer cursor=0;
    protected int sequenceSize ;
    protected int backpropSize ;
    protected Random random ;

    protected int forwardNumber ;
    protected int backpropNumber;
    protected double startTime ;

    public SequentialExperienceReplay(int maxSize, ArrayList<String> file,int sequenceSize, int backpropSize,long seed){
        super(maxSize,file);
        this.resetMemory();
        this.sequenceSize = sequenceSize ;
        this.backpropSize = backpropSize ;
        this.random = new Random(seed);
    }

    @Override
    public void addInteraction(Interaction<A> interaction) {
        if(this.interactions.size() == this.maxSize)
            this.interactions.remove(this.interactions.get(0));
        this.interactions.add(interaction);
    }

    public boolean initChoose(){ // Toujours appeler avant les chooseInteraction
        if(this.interactions.size() == 0)
            return false ;
        if(this.interactions.get(this.interactions.size()-1).getTime() - this.interactions.get(0).getTime() < this.sequenceSize )
            return false ;
        if(this.interactions.size() <= 2)
            return false ;
        // On vérifie qu ele curseur actuel suffit à proposer une séquence complète
        Interaction<A> start = this.choose();
        Double dt = this.interactions.get(this.interactions.size()-1).getTime() - start.getTime() ;
        int cpt = 0 ;
        while(dt < this.sequenceSize || (this.interactions.size() - cursor <= 2)){
            if(cpt == 10){
                cursor=0 ; // On veut limiter le nombre de recherches aléatoires
                break;
            }else {
                start = this.choose();
                dt = this.interactions.get(this.interactions.size()-1).getTime() - start.getTime() ;
                //cursor = 0; // On a déjà vérifié que c'était possible avec 0
                cpt++ ;
            }
        }
        this.startTime = start.getTime() ;
        this.backpropNumber = 0 ;
        this.forwardNumber = 0 ;
        this.tmp = new ArrayList<>();
        return true ;
    }

    protected Interaction<A> choose(){
        cursor = this.random.nextInt(this.interactions.size()-1);
        return this.interactions.get(cursor);
    }

    @Override
    public Interaction<A> chooseInteraction() {
        Interaction<A> choose = this.interactions.get(this.cursor);
        Double dt = choose.getTime() - startTime;
        if(dt > sequenceSize)
            return null;
        this.tmp.add(choose);
        this.forwardNumber++ ;
        this.cursor++ ;
        return choose;
    }

    @Override
    public void resetMemory() {
        this.interactions = new ArrayList<>();
        this.tmp = new ArrayList<>();
    }

    @Override
    public int getSize() {
        return this.interactions.size();
    }

    @Override
    public void setError(INDArray errors) {

    }

    public void setError(INDArray errors, ArrayList<Integer> backpropNumber, int backward,ArrayList<ArrayList<Interaction<A>>> total) {

    }


    public int getForwardNumber(){
        return this.forwardNumber ;
    }

    public int getBackpropNumber(){
        Double endTime = this.tmp.get(this.tmp.size()-1).getTime() ;
        for(int i = this.tmp.size()-1 ; i >= 0 ; i-- ) {
            if (endTime - this.tmp.get(i).getTime() > this.backpropSize) {
                return Math.max(2,Math.min(this.backpropNumber,5));
                //return this.backpropNumber;
            } else
                this.backpropNumber++;
        }
        return Math.max(2,Math.min(this.backpropNumber,5));
        //return this.backpropNumber;
    }

}
