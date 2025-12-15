package pas.tetris.agents;

// SYSTEM IMPORTS
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.util.Random;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

// JAVA PROJECT IMPORTS
import edu.bu.tetris.agents.QAgent;
import edu.bu.tetris.agents.TrainerAgent.GameCounter;
import edu.bu.tetris.game.Game.GameView;
import edu.bu.tetris.game.Game;
import edu.bu.tetris.game.minos.Mino;
import edu.bu.tetris.linalg.Matrix;
import edu.bu.tetris.nn.Model;
import edu.bu.tetris.nn.LossFunction;
import edu.bu.tetris.nn.Optimizer;
import edu.bu.tetris.nn.models.Sequential;
import edu.bu.tetris.nn.layers.Dense;
import edu.bu.tetris.nn.layers.ReLU;
import edu.bu.tetris.training.data.Dataset;
import edu.bu.tetris.utils.Pair;

public class TetrisQAgent extends QAgent {
    private static final double INITIAL_EPSILON = 1.0;
    private static final double MINIMUM_EPSILON = 0.05;
    private static final double EPSILON_DECAY = 0.995;
    private Random random;
    private HashMap<String, Double> rewards;
    private int highestScore = 0;
    private int piecesSinceLastClear = 0;
    private List<Integer> scores = new ArrayList<>();


    public TetrisQAgent(String name) {
        super(name);
        this.random = new Random(12345);
        this.rewards = new HashMap<>();
    }

    public Random getRandom() {
        return this.random;
    }

    @Override
    public Model initQFunction() {
        final int inputDim = 6;
        final int hiddenDim = 2 * inputDim;
        final int outDim = 1;

        Sequential qFunction = new Sequential();
        qFunction.add(new Dense(inputDim, hiddenDim));
        qFunction.add(new ReLU());
        qFunction.add(new Dense(hiddenDim, outDim));

        return qFunction;
    }

    /* MATRIX FLATTENING WITH 6 FEATURES IMPLEMENTED AND REWARD FUNCTION */

    @Override
    public Matrix getQFunctionInput(final GameView game, final Mino potentialAction) {
        if (game == null || potentialAction == null) {
            return null;
        }

        Matrix qVectorInput = Matrix.full(1, 6, 0.0);
        try {
            Matrix grayscale = game.getGrayscaleImage(potentialAction);
            return calculateFeatures(grayscale, qVectorInput);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }
        return qVectorInput;
    }

    private Matrix calculateFeatures(Matrix grayscale, Matrix qVectorInput) {
        int cols = grayscale.getShape().getNumCols();
        HashMap<Integer, ArrayList<Double>> columnInfo = extractColumnInfo(grayscale);
        ArrayList<Integer> vertical = new ArrayList<>();
        boolean[] completeLine = new boolean[cols];
        Arrays.fill(completeLine, true);
        calculateColumnFeatures(columnInfo, vertical, completeLine, qVectorInput);
        return qVectorInput;
    }

    private HashMap<Integer, ArrayList<Double>> extractColumnInfo(Matrix grayscale) {
        int rows = grayscale.getShape().getNumRows();
        int cols = grayscale.getShape().getNumCols();
        HashMap<Integer, ArrayList<Double>> columnInfo = new HashMap<>();

        for (int y = 0; y < cols; y++) {
            ArrayList<Double> columnData = new ArrayList<>();
            for (int x = 2; x < rows; x++) {
                columnData.add(grayscale.get(x, y));
            }
            columnInfo.put(y, columnData);
        }

        return columnInfo;
    }

    private void calculateColumnFeatures(HashMap<Integer, ArrayList<Double>> columnInfo, ArrayList<Integer> vertical, boolean[] completeLine, Matrix qVectorInput) {
        // NOTE TO SELF - 0: holes, 1: botomMino, 2: columnHoles
        int[] stats = new int[3];
        int completedLines = 0;
    
        for (Map.Entry<Integer, ArrayList<Double>> entry : columnInfo.entrySet()) {
            int maxHeight = calculateMaxHeight(entry.getValue(), stats);
            vertical.add(maxHeight);
            completeLine[entry.getKey()] = isLineComplete(entry.getValue());
        }
    
        for (boolean lineComplete : completeLine) {
            if (lineComplete) completedLines++;
        }
    
        setFeatureValues(qVectorInput, stats[0], stats[2], vertical, stats[1], completedLines);
    }
    
    private boolean isLineComplete(List<Double> column) {
        return column.stream().noneMatch(val -> val == Game.GameView.UNOCCUPIED_COORDINATE_VALUE);
    }
    
    private int calculateMaxHeight(List<Double> column, int[] stats) {
        int maxHeight = 0;  
        boolean foundBlock = false;
        int bottomMino = stats[1];
    
        for (int i = 0; i < column.size(); i++) {
            double value = column.get(i);
            if (value != Game.GameView.UNOCCUPIED_COORDINATE_VALUE) {
                foundBlock = true;
                maxHeight = column.size() - i;
                bottomMino = Math.max(bottomMino, i);
                break;
            }
        }
    
        if (foundBlock) {
            for (int i = column.size() - maxHeight; i < column.size(); i++) {
                if (column.get(i) == Game.GameView.UNOCCUPIED_COORDINATE_VALUE) {
                    stats[0]++;
                    if (i == column.size() - maxHeight) {
                        stats[2]++;
                    }
                }
            }
        }
    
        stats[1] = bottomMino;
        return maxHeight;
    }

    private double calculateBumpiness(List<Integer> heights) {
        double sumOfHeights = heights.stream().mapToInt(Integer::intValue).sum();
        double mean = sumOfHeights / heights.size();
        double bumpiness = heights.stream()
                                  .mapToDouble(h -> Math.pow(h - mean, 2))
                                  .sum();
        return Math.sqrt(bumpiness / heights.size());
    }

    private void setFeatureValues(Matrix qVectorInput, int holes, double columnHoles, List<Integer> vertical, int bottomMino, int completedLines) {
        double bumpiness = calculateBumpiness(vertical);
        double verticalDifference = Collections.max(vertical) - Collections.min(vertical);

        qVectorInput.set(0, 0, holes);
        qVectorInput.set(0, 1, bumpiness);
        qVectorInput.set(0, 2, verticalDifference);
        qVectorInput.set(0, 3, columnHoles);
        qVectorInput.set(0, 4, bottomMino);
        qVectorInput.set(0, 5, completedLines);

        this.rewards.put("Holes", (double) holes);
        this.rewards.put("VerticalStats", verticalDifference);
        this.rewards.put("Bumpiness", bumpiness);
        this.rewards.put("ColumnHoles", columnHoles);
        this.rewards.put("BottomMino", 1.0 * bottomMino);
        this.rewards.put("CompletedLines", (double) completedLines);
    }

    /* REWARD ARCHITECTURE */

    @Override
    public double getReward(final GameView game) {
        if (game == null || game.getBoard() == null) {
            return 0;
        }

        double reward = 0;
        reward += calculateBaseReward(game);
        reward += calculateBonusRewards(game);
        reward -= deductPenalties(game);

        int currentScore = game.getTotalScore();
        recordScore(currentScore);
        if (currentScore > highestScore) {
            highestScore = currentScore;
            reward += 500;
        }

        if (game.didAgentLose()) {
            reward -= 1000;
        }

        return reward;
    }

    private double calculateBaseReward(GameView game) {
        return 1000 * game.getScoreThisTurn(); 
    }

    private double calculateBonusRewards(GameView game) {
        double completedLines = this.rewards.getOrDefault("CompletedLines", 0.0);
        double reward = 0;

        if (completedLines > 0) {
            double efficiencyMultiplier = Math.max(0.5, 2.0 - (0.2 * piecesSinceLastClear));
            switch ((int) completedLines) {
                case 1:
                    reward = 150 * efficiencyMultiplier;
                    break;
                case 2:
                    reward = 350 * efficiencyMultiplier;
                    break;
                case 3:
                    reward = 800 * efficiencyMultiplier;
                    break;
                case 4:
                    reward = 1100 * efficiencyMultiplier;
                    break;
                default:
                    reward = 150 * completedLines * efficiencyMultiplier;
                    break;
            }
            piecesSinceLastClear = 0;
        } else {
            piecesSinceLastClear++;
        }
        
        return reward;
    }
    
    private double deductPenalties(GameView game) {
        double penalty = 0;
        penalty += 0.75 * this.rewards.getOrDefault("Holes", 0.0);
        penalty += 2.0 * this.rewards.getOrDefault("Bumpiness", 0.0);  
        penalty += 1.5 * this.rewards.getOrDefault("VerticalStats", 0.0); 
        penalty += 10 * this.rewards.getOrDefault("ColumnHoles", 0.0); 
        penalty += 0.5 * this.rewards.getOrDefault("BottomMino", 0.0);

        return penalty;
    }

    /* EXPLORATION IMPLEMENTATION */

    @Override
    public boolean shouldExplore(final GameView game, final GameCounter gameCounter) {
        double baseEpsilon = Math.max(MINIMUM_EPSILON, INITIAL_EPSILON * Math.pow(EPSILON_DECAY, gameCounter.getTotalGamesPlayed()));
        double scoreImprovementFactor = calculateScoreImprovementFactor(game, gameCounter);
        double epsilon = baseEpsilon * scoreImprovementFactor;
        return getRandom().nextDouble() < epsilon;
    }

    private double calculateScoreImprovementFactor(GameView game, GameCounter gameCounter) {
        double scoreImprovementFactor = 1.0;
    
        if (gameCounter.getCurrentGameIdx() > 1) { 
            int lastScore = getLastGameScore();
            int currentScore = game.getTotalScore();
    
            if (currentScore > lastScore) {
                scoreImprovementFactor = 0.9;
            } else if (currentScore < lastScore) {
                scoreImprovementFactor = 1.1;
            }
        }
        return scoreImprovementFactor;
    }

    public void recordScore(int score) {
        scores.add(score);
    }

    private int getLastGameScore() {
        if (scores.size() > 1) {
            return scores.get(scores.size() - 2);
        }
        return 0;
    }

    @Override
    public Mino getExplorationMove(final GameView game) {
        List<Mino> positions = game.getFinalMinoPositions();
        if (positions.isEmpty()) {
            return null;
        }

        for (Mino position : positions) {
            if (isPotentialTetris(game, position)) {
                return position;
            }
        }

        return positions.get(this.getRandom().nextInt(positions.size()));
    }

    private boolean isPotentialTetris(GameView game, Mino position) {
        Matrix grayscale = null;
        try{
            grayscale = game.getGrayscaleImage(position);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }
        
        int rows = grayscale.getShape().getNumRows();
        ArrayList<Double> blockInfo = new ArrayList<>();

        for (int row = 2; row < rows; row++) {
            blockInfo.add(grayscale.get(row, 0));
        }

        Collections.reverse(blockInfo);
        int count = 0;

        for (int j = 0; j < Math.min(6, blockInfo.size()); j++) {
            if (blockInfo.get(j) == 1.0 || blockInfo.get(j) == 0.5) {
                count++;
            }
        }

        return count >= 4;
    }

    @Override
    public void trainQFunction(Dataset dataset, LossFunction lossFunction, Optimizer optimizer, long numUpdates) {
        for(int epochIdx = 0; epochIdx < numUpdates; ++epochIdx) {
            dataset.shuffle();
            Iterator<Pair<Matrix, Matrix> > batchIterator = dataset.iterator();

            while(batchIterator.hasNext()) {
                Pair<Matrix, Matrix> batch = batchIterator.next();
                try {
                    Matrix YHat = this.getQFunction().forward(batch.getFirst());
                    optimizer.reset();
                    this.getQFunction().backwards(batch.getFirst(), lossFunction.backwards(YHat, batch.getSecond()));
                    optimizer.step();
                } catch(Exception e) {
                    e.printStackTrace();
                    System.exit(-1);
                }
            }
        }
    }
}