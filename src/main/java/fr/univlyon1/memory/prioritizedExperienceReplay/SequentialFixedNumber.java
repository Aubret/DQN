package fr.univlyon1.memory.prioritizedExperienceReplay;

import fr.univlyon1.environment.Interaction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;

public class SequentialFixedNumber<A> extends SequentialPrioritizedExperienceReplay<A> {
    protected int endCursor ;

    public SequentialFixedNumber(int maxSize, ArrayList<String> file, int sequenceSize, int backpropSize, long seed, int learn) {
        super(maxSize, file, sequenceSize, backpropSize, seed, learn);
        this.minForward = sequenceSize;
    }

    protected Interaction<A> choose() {
        Interaction<A> start = this.prioritized.chooseInteraction();
        this.cursor = this.interactions.indexOf(start);
        return start;
    }

    @Override
    public Interaction<A> chooseInteraction() {
        if(this.cursor > this.endCursor)
            return null;
        Interaction<A> choose = this.interactions.get(this.cursor);
        this.tmp.add(choose);
        this.forwardNumber++ ;
        this.cursor++ ;
        return choose;
    }

    public boolean initChoose() { // Toujours appeler avant les chooseInteraction
        if (this.prioritized.getSize() == 0)
            return false;
        if (this.interactions.size() <= minForward)
            return false;

        // On vérifie qu ele curseur actuel suffit à proposer une séquence complète
        this.choose();
        int cpt = 0;
        while (this.interactions.size() - cursor <= minForward) {
            if (cpt == 10) {
                this.prioritized.repushLast();// On replace le dernier choisi
                cursor = 0; // On veut limiter le nombre de recherches aléatoires
                break;
            } else {
                this.prioritized.repushLast();// On replace le dernier choisi
                this.choose();
                //cursor = 0; // On a déjà vérifié que c'était possible avec 0
                cpt++;
            }
        }
        this.endCursor = cursor + minForward-1 ;
        this.backpropNumber = 0;
        this.forwardNumber = 0;
        this.tmp = new ArrayList<>();
        return true;
    }

    public boolean isAvailable(Interaction<A> interaction) {
        int curs = this.interactions.indexOf(interaction);
        if (this.interactions.size() - curs <= minForward)
            return false;
        return true;
    }

    public int getBackpropNumber() {
        return 2; // 2 - 1 = une seule interaction par batch, on sauvegarde donc la dernière donnée en mémoire
    }
}