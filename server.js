const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const { Chess } = require('chess.js');
const tf = require('@tensorflow/tfjs-node');

const app = express();
const port = 3000;

app.use(cors());
app.use(express.json());
app.use(express.static('public'));

function isGameOver(chess) {
    try {
        if (typeof chess.isGameOver === 'function') {
            return chess.isGameOver();
        } else if (typeof chess.game_over === 'function') {
            return chess.game_over();
        } else {
            return chess.isCheckmate() || chess.isStalemate() || chess.isDraw();
        }
    } catch (error) {
        try {
            return chess.in_checkmate() || chess.in_stalemate() || chess.in_draw();
        } catch (legacyError) {
            console.error('Cannot determine game over status:', error);
            return false;
        }
    }
}

function isCheckmate(chess) {
    try {
        if (typeof chess.isCheckmate === 'function') {
            return chess.isCheckmate();
        } else if (typeof chess.in_checkmate === 'function') {
            return chess.in_checkmate();
        }
    } catch (error) {
        console.error('Cannot determine checkmate:', error);
    }
    return false;
}

function getGameResult(chess) {
    try {
        if (isCheckmate(chess)) {
            return chess.turn() === 'w' ? "Black wins by checkmate!" : "White wins by checkmate!";
        }
        if (typeof chess.isStalemate === 'function' && chess.isStalemate()) {
            return "Draw by stalemate";
        }
        if (typeof chess.in_stalemate === 'function' && chess.in_stalemate()) {
            return "Draw by stalemate";
        }
        return "Draw";
    } catch (error) {
        console.error('Error determining game result:', error);
        return "Game Over";
    }
}

function createChessGame(fen) {
    try {
        const chess = new Chess();
        const startingFEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
        
        if (!fen || fen === startingFEN) {
            return { success: true, game: chess, message: 'Starting position loaded' };
        }
        
        try {
            chess.load(fen);
            const moves = chess.moves();
            return { 
                success: true, 
                game: chess, 
                message: 'Position loaded successfully, ' + moves.length + ' moves available'
            };
        } catch (loadError) {
            return { 
                success: false, 
                error: 'Invalid FEN position: ' + loadError.message 
            };
        }
    } catch (error) {
        return { 
            success: false, 
            error: 'Failed to create chess game: ' + error.message 
        };
    }
}

class ChessAI {
    constructor() {
        this.model = null;
        this.generation = 0;
        this.gamesPlayed = 0;
        this.wins = 0;
        this.losses = 0;
        this.draws = 0;
        this.trainingData = [];
        this.gameHistory = [];
        
        this.modelPath = './chess_model';
        this.metadataPath = './chess_metadata.json';
        this.trainingDataPath = './chess_training_data.json';
        this.gameHistoryPath = './chess_game_history.json';
        
        this.learningRate = 0.001;
        this.gamma = 0.95;
        this.epsilon = 0.1;
        this.batchSize = 32;
        this.maxTrainingData = 1000000;
        this.maxGameHistory = 1000;
        
        this.autoReset = false;
        this.maxGenerations = Infinity;
        this.isResetting = false;
        
        this.initializeModel();
        this.loadAllData();
    }
    
    initializeModel() {
        this.model = tf.sequential({
            layers: [
                tf.layers.dense({ inputShape: [773], units: 512, activation: 'relu' }),
                tf.layers.dropout({ rate: 0.3 }),
                tf.layers.dense({ units: 256, activation: 'relu' }),
                tf.layers.dropout({ rate: 0.3 }),
                tf.layers.dense({ units: 128, activation: 'relu' }),
                tf.layers.dense({ units: 1, activation: 'tanh' })
            ]
        });
        
        this.model.compile({
            optimizer: tf.train.adam(this.learningRate),
            loss: 'meanSquaredError',
            metrics: ['mae']
        });
    }
    
    positionToFeatures(chess) {
        const features = new Float32Array(773);
        let idx = 0;
        
        const board = chess.board();
        for (let i = 0; i < 8; i++) {
            for (let j = 0; j < 8; j++) {
                const piece = board[i][j];
                if (piece) {
                    const pieceIdx = this.getPieceIndex(piece.type, piece.color);
                    features[idx + pieceIdx] = 1;
                }
                idx += 12;
            }
        }
        
        features[768] = chess.turn() === 'w' ? 1 : -1;
        features[769] = chess.inCheck() ? 1 : 0;
        features[770] = chess.moves().length / 50;
        features[771] = chess.history().length / 100;
        features[772] = this.getMaterialBalance(chess) / 39;
        
        return features;
    }
    
    getPieceIndex(type, color) {
        const pieces = { 'p': 0, 'r': 1, 'n': 2, 'b': 3, 'q': 4, 'k': 5 };
        return pieces[type] + (color === 'w' ? 0 : 6);
    }
    
    getMaterialBalance(chess) {
        const values = { 'p': 1, 'r': 5, 'n': 3, 'b': 3, 'q': 9, 'k': 0 };
        let balance = 0;
        
        const board = chess.board();
        for (let i = 0; i < 8; i++) {
            for (let j = 0; j < 8; j++) {
                const piece = board[i][j];
                if (piece) {
                    const value = values[piece.type];
                    balance += piece.color === 'w' ? value : -value;
                }
            }
        }
        return balance;
    }
    
    async evaluatePosition(chess) {
        const features = this.positionToFeatures(chess);
        const tensorInput = tf.tensor2d([features]);
        const prediction = await this.model.predict(tensorInput);
        const value = await prediction.data();
        
        tensorInput.dispose();
        prediction.dispose();
        
        return value[0];
    }
    
    async selectMove(chess, depth) {
        if (!depth) depth = 3;
        
        const moves = chess.moves();
        if (moves.length === 0) return null;
        
        if (Math.random() < this.epsilon) {
            return moves[Math.floor(Math.random() * moves.length)];
        }
        
        let bestMove = moves[0];
        let bestValue = chess.turn() === 'w' ? -Infinity : Infinity;
        
        for (const move of moves) {
            const chessCopy = new Chess(chess.fen());
            chessCopy.move(move);
            
            const value = await this.minimax(chessCopy, depth - 1, chess.turn() === 'b', -Infinity, Infinity);
            
            if (chess.turn() === 'w' && value > bestValue) {
                bestValue = value;
                bestMove = move;
            } else if (chess.turn() === 'b' && value < bestValue) {
                bestValue = value;
                bestMove = move;
            }
        }
        
        return bestMove;
    }
    
    async minimax(chess, depth, maximizing, alpha, beta) {
        if (depth === 0 || isGameOver(chess)) {
            if (isGameOver(chess)) {
                if (isCheckmate(chess)) {
                    return maximizing ? -1000 : 1000;
                }
                return 0;
            }
            return await this.evaluatePosition(chess);
        }
        
        const moves = chess.moves();
        
        if (maximizing) {
            let maxEval = -Infinity;
            for (const move of moves) {
                const chessCopy = new Chess(chess.fen());
                chessCopy.move(move);
                const evaluation = await this.minimax(chessCopy, depth - 1, false, alpha, beta);
                maxEval = Math.max(maxEval, evaluation);
                alpha = Math.max(alpha, evaluation);
                if (beta <= alpha) break;
            }
            return maxEval;
        } else {
            let minEval = Infinity;
            for (const move of moves) {
                const chessCopy = new Chess(chess.fen());
                chessCopy.move(move);
                const evaluation = await this.minimax(chessCopy, depth - 1, true, alpha, beta);
                minEval = Math.min(minEval, evaluation);
                beta = Math.min(beta, evaluation);
                if (beta <= alpha) break;
            }
            return minEval;
        }
    }
    
    storeGameData(positions, result, gameInfo) {
        if (!gameInfo) gameInfo = {};
        
        const gameData = {
            id: Date.now() + Math.random(),
            timestamp: new Date().toISOString(),
            result: result,
            totalMoves: positions.length,
            gameInfo: gameInfo
        };
        
        this.gameHistory.push(gameData);
        
        for (let i = 0; i < positions.length; i++) {
            const reward = this.calculateReward(result, i, positions.length);
            this.trainingData.push({
                gameId: gameData.id,
                moveIndex: i,
                features: positions[i],
                reward: reward,
                timestamp: gameData.timestamp
            });
        }
        
        if (this.trainingData.length > this.maxTrainingData) {
            this.trainingData = this.trainingData.slice(-this.maxTrainingData);
        }
        
        if (this.gameHistory.length > this.maxGameHistory) {
            this.gameHistory = this.gameHistory.slice(-this.maxGameHistory);
        }
        
        this.saveTrainingData();
        this.saveGameHistory();
    }
    
    calculateReward(result, moveIndex, totalMoves) {
        let baseReward = 0;
        
        if (result === 'win') baseReward = 1;
        else if (result === 'loss') baseReward = -1;
        else baseReward = 0;
        
        const progressFactor = moveIndex / totalMoves;
        return baseReward * (0.5 + 0.5 * progressFactor);
    }
    
    async trainModel() {
        if (this.trainingData.length < this.batchSize) return;
        
        console.log('Training model with ' + this.trainingData.length + ' samples...');
        
        const features = this.trainingData.map(d => d.features);
        const rewards = this.trainingData.map(d => d.reward);
        
        const xs = tf.tensor2d(features);
        const ys = tf.tensor2d(rewards, [rewards.length, 1]);
        
        await this.model.fit(xs, ys, {
            epochs: 10,
            batchSize: this.batchSize,
            verbose: 0
        });
        
        xs.dispose();
        ys.dispose();
        
        console.log('Model training completed');
    }
    
    saveTrainingData() {
        try {
            const dataToSave = this.trainingData.map(d => ({
                gameId: d.gameId,
                moveIndex: d.moveIndex,
                features: Array.from(d.features),
                reward: d.reward,
                timestamp: d.timestamp
            }));
            
            fs.writeFileSync(this.trainingDataPath, JSON.stringify(dataToSave, null, 2));
            console.log('Training data saved: ' + this.trainingData.length + ' samples');
        } catch (error) {
            console.error('Error saving training data:', error);
        }
    }
    
    loadTrainingData() {
        try {
            if (fs.existsSync(this.trainingDataPath)) {
                const data = JSON.parse(fs.readFileSync(this.trainingDataPath, 'utf8'));
                this.trainingData = data.map(d => ({
                    gameId: d.gameId,
                    moveIndex: d.moveIndex,
                    features: new Float32Array(d.features),
                    reward: d.reward,
                    timestamp: d.timestamp
                }));
                console.log('Training data loaded: ' + this.trainingData.length + ' samples');
            } else {
                console.log('No existing training data found');
            }
        } catch (error) {
            console.error('Error loading training data:', error);
            this.trainingData = [];
        }
    }
    
    saveGameHistory() {
        try {
            fs.writeFileSync(this.gameHistoryPath, JSON.stringify(this.gameHistory, null, 2));
            console.log('Game history saved: ' + this.gameHistory.length + ' games');
        } catch (error) {
            console.error('Error saving game history:', error);
        }
    }
    
    loadGameHistory() {
        try {
            if (fs.existsSync(this.gameHistoryPath)) {
                this.gameHistory = JSON.parse(fs.readFileSync(this.gameHistoryPath, 'utf8'));
                console.log('Game history loaded: ' + this.gameHistory.length + ' games');
            } else {
                console.log('No existing game history found');
            }
        } catch (error) {
            console.error('Error loading game history:', error);
            this.gameHistory = [];
        }
    }
    
    async saveModel() {
        try {
            await this.model.save('file://' + this.modelPath);
            
            const metadata = {
                generation: this.generation,
                gamesPlayed: this.gamesPlayed,
                wins: this.wins,
                losses: this.losses,
                draws: this.draws,
                winRate: this.gamesPlayed > 0 ? (this.wins / this.gamesPlayed * 100).toFixed(2) : 0,
                trainingDataSize: this.trainingData.length,
                gameHistorySize: this.gameHistory.length,
                autoReset: this.autoReset,
                maxGenerations: this.maxGenerations,
                epsilon: this.epsilon,
                lastSaved: new Date().toISOString()
            };
            
            fs.writeFileSync(this.metadataPath, JSON.stringify(metadata, null, 2));
            console.log('Model saved - Generation ' + this.generation + ', Games: ' + this.gamesPlayed + ', Win Rate: ' + metadata.winRate + '%');
        } catch (error) {
            console.error('Error saving model:', error);
        }
    }
    
    async loadModel() {
        try {
            if (fs.existsSync(this.modelPath)) {
                this.model = await tf.loadLayersModel('file://' + this.modelPath + '/model.json');
                
                this.model.compile({
                    optimizer: tf.train.adam(this.learningRate),
                    loss: 'meanSquaredError',
                    metrics: ['mae']
                });
                
                console.log('Existing model loaded and compiled');
            }
            
            if (fs.existsSync(this.metadataPath)) {
                const metadata = JSON.parse(fs.readFileSync(this.metadataPath, 'utf8'));
                this.generation = metadata.generation || 0;
                this.gamesPlayed = metadata.gamesPlayed || 0;
                this.wins = metadata.wins || 0;
                this.losses = metadata.losses || 0;
                this.draws = metadata.draws || 0;
                this.epsilon = metadata.epsilon || 0.1;
                
                console.log('Loaded metadata - Generation ' + this.generation + ', Games: ' + this.gamesPlayed + ', Epsilon: ' + this.epsilon);
            }
        } catch (error) {
            console.log('No existing model found, starting fresh');
        }
    }
    
    async loadAllData() {
        await this.loadModel();
        this.loadTrainingData();
        this.loadGameHistory();
        console.log('Total data loaded - Training samples: ' + this.trainingData.length + ', Game history: ' + this.gameHistory.length);
    }
    
    async saveAllData() {
        await this.saveModel();
        this.saveTrainingData();
        this.saveGameHistory();
    }
    
    async nextGeneration() {
        await this.trainModel();
        this.generation++;
        
        this.epsilon = Math.max(0.01, this.epsilon * 0.995);
        
        await this.saveAllData();
        console.log('Advanced to generation ' + this.generation);
    }
    
    async processGameEnd(gameResult, gamePositions, gameInfo) {
        if (!gameInfo) gameInfo = {};
        
        if (gameResult === 'win') this.wins++;
        else if (gameResult === 'loss') this.losses++;
        else this.draws++;
        
        this.gamesPlayed++;
        
        const extendedGameInfo = {
            gameNumber: this.gamesPlayed,
            generation: this.generation
        };
        
        Object.assign(extendedGameInfo, gameInfo);
        
        this.storeGameData(gamePositions, gameResult, extendedGameInfo);
        
        console.log('Game ' + this.gamesPlayed + ' completed: ' + gameResult);
        
        if (this.gamesPlayed % 10 === 0) {
            console.log('Training after ' + this.gamesPlayed + ' games...');
            await this.nextGeneration();
        } else {
            await this.saveAllData();
        }
        
        return false;
    }
    
    getStats() {
        return {
            generation: this.generation,
            gamesPlayed: this.gamesPlayed,
            wins: this.wins,
            losses: this.losses,
            draws: this.draws,
            winRate: this.gamesPlayed > 0 ? (this.wins / this.gamesPlayed * 100).toFixed(2) : 0,
            epsilon: this.epsilon.toFixed(3),
            trainingDataSize: this.trainingData.length,
            gameHistorySize: this.gameHistory.length,
            autoReset: false,
            continuousLearning: true,
            dataStorageEnabled: true,
            lastGame: this.gameHistory.length > 0 ? this.gameHistory[this.gameHistory.length - 1] : null
        };
    }
    
    getGameHistory(limit) {
        if (!limit) limit = 10;
        return this.gameHistory.slice(-limit).reverse();
    }
    
    exportTrainingData() {
        return {
            metadata: {
                totalSamples: this.trainingData.length,
                generations: this.generation,
                gamesPlayed: this.gamesPlayed,
                exportDate: new Date().toISOString()
            },
            trainingData: this.trainingData.slice(-1000),
            gameHistory: this.gameHistory.slice(-100)
        };
    }
    
    async resetTraining() {
        console.log('Resetting AI training...');
        
        if (this.model) {
            this.model.dispose();
            this.model = null;
        }
        
        this.generation = 0;
        this.gamesPlayed = 0;
        this.wins = 0;
        this.losses = 0;
        this.draws = 0;
        this.trainingData = [];
        this.gameHistory = [];
        this.epsilon = 0.1;
        
        try {
            if (fs.existsSync(this.modelPath)) {
                fs.rmSync(this.modelPath, { recursive: true, force: true });
            }
            if (fs.existsSync(this.metadataPath)) {
                fs.unlinkSync(this.metadataPath);
            }
            if (fs.existsSync(this.trainingDataPath)) {
                fs.unlinkSync(this.trainingDataPath);
            }
            if (fs.existsSync(this.gameHistoryPath)) {
                fs.unlinkSync(this.gameHistoryPath);
            }
        } catch (error) {
            console.error('Error deleting files:', error);
        }
        
        this.initializeModel();
        await this.saveAllData();
        
        console.log('Training has been reset to Generation 0. All data cleared.');
    }
    
    async manualReset() {
        this.isResetting = true;
        try {
            await this.resetTraining();
        } finally {
            this.isResetting = false;
        }
    }
}

const chessAI = new ChessAI();
let currentGame = new Chess();
let gamePositions = [];

app.post('/move', async (req, res) => {
    try {
        if (isGameOver(currentGame)) {
            return res.json({ error: 'Game is over' });
        }
        
        gamePositions.push(chessAI.positionToFeatures(currentGame));
        
        const move = await chessAI.selectMove(currentGame);
        
        if (!move || !currentGame.move(move)) {
            return res.json({ error: 'Invalid move generated' });
        }
        
        let gameResult = null;
        let shouldStartNewGame = false;
        
        if (isGameOver(currentGame)) {
            if (isCheckmate(currentGame)) {
                gameResult = currentGame.turn() === 'w' ? 'loss' : 'win';
            } else {
                gameResult = 'draw';
            }
            
            const gameInfo = {
                moveCount: gamePositions.length,
                fen: currentGame.fen(),
                pgn: currentGame.pgn()
            };
            
            await chessAI.processGameEnd(gameResult, gamePositions, gameInfo);
            
            currentGame = new Chess();
            gamePositions = [];
            shouldStartNewGame = true;
            
            console.log('Game completed: ' + gameResult + '. Starting new game automatically...');
        }
        
        res.json({
            fen: currentGame.fen(),
            move: move,
            isGameOver: !shouldStartNewGame,
            result: gameResult ? getGameResult(currentGame) : null,
            aiStats: chessAI.getStats(),
            newGameStarted: shouldStartNewGame
        });
        
    } catch (error) {
        console.error('Error processing move:', error);
        res.status(500).json({ error: error.message });
    }
});

app.post('/reset', async (req, res) => {
    currentGame = new Chess();
    gamePositions = [];
    res.json({ 
        fen: currentGame.fen(),
        aiStats: chessAI.getStats()
    });
});

app.get('/stats', (req, res) => {
    res.json(chessAI.getStats());
});

app.get('/game-history', (req, res) => {
    const limit = parseInt(req.query.limit) || 10;
    res.json(chessAI.getGameHistory(limit));
});

app.get('/export-data', (req, res) => {
    const exportData = chessAI.exportTrainingData();
    res.json(exportData);
});

app.post('/force-training', async (req, res) => {
    try {
        await chessAI.nextGeneration();
        res.json({ 
            message: 'Training completed',
            stats: chessAI.getStats()
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.post('/manual-reset', async (req, res) => {
    try {
        await chessAI.manualReset();
        currentGame = new Chess();
        gamePositions = [];
        res.json({ 
            message: 'AI has been manually reset - all data cleared',
            stats: chessAI.getStats()
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.post('/save-data', async (req, res) => {
    try {
        await chessAI.saveAllData();
        res.json({ 
            message: 'All data saved successfully',
            stats: chessAI.getStats()
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', '*');
    res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
    res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization');
    
    if (req.method === 'OPTIONS') {
        res.sendStatus(200);
    } else {
        next();
    }
});

app.get('/player-test', (req, res) => {
    try {
        console.log('=== CHESS DIAGNOSTICS ===');
        
        const testGame = new Chess();
        console.log('Chess game created');
        
        const methods = {
            fen: typeof testGame.fen === 'function',
            moves: typeof testGame.moves === 'function',
            move: typeof testGame.move === 'function',
            load: typeof testGame.load === 'function',
            isGameOver: typeof testGame.isGameOver === 'function',
            game_over: typeof testGame.game_over === 'function',
            in_checkmate: typeof testGame.in_checkmate === 'function',
            in_stalemate: typeof testGame.in_stalemate === 'function',
            in_draw: typeof testGame.in_draw === 'function',
            turn: typeof testGame.turn === 'function'
        };
        console.log('Available methods:', methods);
        
        const startFen = testGame.fen();
        const startMoves = testGame.moves();
        console.log('Starting position:', startFen);
        console.log('Starting moves:', startMoves.length);
        
        const testPosition = 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1';
        const loadResult = createChessGame(testPosition);
        console.log('Position load test:', loadResult.success ? 'PASS' : 'FAIL');
        if (!loadResult.success) {
            console.log('Load error:', loadResult.error);
        }
        
        const aiAvailable = typeof chessAI !== 'undefined' && typeof chessAI.selectMove === 'function';
        console.log('AI available:', aiAvailable);
        
        console.log('=== END DIAGNOSTICS ===');
        
        res.json({ 
            success: true,
            message: 'Training server diagnostics complete',
            timestamp: new Date().toISOString(),
            diagnostics: {
                chessVersion: 'detected',
                methods: methods,
                startingPosition: startFen,
                startingMoves: startMoves.length,
                positionLoadTest: loadResult.success,
                aiAvailable: aiAvailable,
                aiGeneration: chessAI.generation,
                aiGamesPlayed: chessAI.gamesPlayed,
                aiWinRate: chessAI.gamesPlayed > 0 ? (chessAI.wins / chessAI.gamesPlayed * 100).toFixed(2) : 0
            }
        });
        
    } catch (error) {
        console.error('Test endpoint error:', error);
        res.status(500).json({
            success: false,
            message: 'Diagnostics failed',
            error: error.message,
            timestamp: new Date().toISOString()
        });
    }
});

app.post('/player-move', async (req, res) => {
    try {
        const { fen } = req.body;
        
        console.log('=== PLAYER VS AI MOVE REQUEST ===');
        console.log('Received FEN:', fen);
        
        if (!fen) {
            return res.status(400).json({ error: 'FEN position required' });
        }
        
        const gameResult = createChessGame(fen);
        if (!gameResult.success) {
            console.error('Failed to create/load chess position:', gameResult.error);
            return res.status(400).json({ error: gameResult.error });
        }
        
        const tempGame = gameResult.game;
        console.log('Chess position loaded successfully');
        console.log('Current FEN:', tempGame.fen());
        console.log('Current turn:', tempGame.turn());
        
        if (isGameOver(tempGame)) {
            console.log('Game is already over');
            return res.status(400).json({ error: 'Game is already over' });
        }
        
        const availableMoves = tempGame.moves();
        console.log('Available moves for AI:', availableMoves.length);
        console.log('Sample moves:', availableMoves.slice(0, 5));
        
        if (availableMoves.length === 0) {
            return res.status(400).json({ error: 'No legal moves available' });
        }
        
        console.log('Requesting move from trained AI (Gen ' + chessAI.generation + ')...');
        
        let aiMove;
        try {
            aiMove = await chessAI.selectMove(tempGame, 3);
        } catch (error) {
            console.error('AI selectMove failed:', error);
            return res.status(500).json({ error: 'AI move selection failed: ' + error.message });
        }
        
        if (!aiMove) {
            console.error('AI returned null/undefined move');
            return res.status(500).json({ error: 'AI could not find a valid move' });
        }
        
        console.log('AI selected move:', aiMove);
        
        let moveResult;
        try {
            moveResult = tempGame.move(aiMove);
        } catch (error) {
            console.error('Error applying AI move:', error);
            return res.status(500).json({ error: 'Error applying AI move: ' + error.message });
        }
        
        if (!moveResult) {
            console.error('Move was rejected by chess.js:', aiMove);
            return res.status(500).json({ error: 'Invalid move generated by AI: ' + aiMove });
        }
        
        console.log('AI move applied:', moveResult.san);
        console.log('New position FEN:', tempGame.fen());
        
        const gameOverAfterMove = isGameOver(tempGame);
        console.log('Game over after AI move:', gameOverAfterMove);
        
        console.log('=== MOVE REQUEST COMPLETE ===');
        
        res.json({
            success: true,
            move: aiMove,
            san: moveResult.san,
            from: moveResult.from,
            to: moveResult.to,
            fen: tempGame.fen(),
            isGameOver: gameOverAfterMove,
            result: gameOverAfterMove ? getGameResult(tempGame) : null,
            aiInfo: {
                generation: chessAI.generation,
                gamesPlayed: chessAI.gamesPlayed,
                winRate: chessAI.gamesPlayed > 0 ? (chessAI.wins / chessAI.gamesPlayed * 100).toFixed(2) : 0
            }
        });
        
    } catch (error) {
        console.error('=== PLAYER MOVE ENDPOINT ERROR ===');
        console.error(error);
        res.status(500).json({ error: 'Server error: ' + error.message });
    }
});

app.post('/player-evaluate', async (req, res) => {
    try {
        const { fen } = req.body;
        
        if (!fen) {
            return res.status(400).json({ error: 'FEN position required' });
        }
        
        const gameResult = createChessGame(fen);
        if (!gameResult.success) {
            return res.status(400).json({ error: gameResult.error });
        }
        
        const tempGame = gameResult.game;
        const evaluation = await chessAI.evaluatePosition(tempGame);
        
        res.json({
            success: true,
            evaluation: evaluation,
            perspective: tempGame.turn() === 'w' ? 'white' : 'black',
            aiInfo: {
                generation: chessAI.generation,
                gamesPlayed: chessAI.gamesPlayed
            }
        });
        
    } catch (error) {
        console.error('Error evaluating position:', error);
        res.status(500).json({ error: 'Evaluation error: ' + error.message });
    }
});

app.get('/run-tests', async (req, res) => {
    const testResults = [];
    
    try {
        console.log('=== RUNNING UNIT TESTS ===');
        
        try {
            const game = new Chess();
            const fen = game.fen();
            testResults.push({
                test: 'Chess game creation',
                passed: fen === 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
                result: fen
            });
        } catch (error) {
            testResults.push({
                test: 'Chess game creation',
                passed: false,
                error: error.message
            });
        }
        
        try {
            const testFEN = 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1';
            const result = createChessGame(testFEN);
            testResults.push({
                test: 'Position loading (after e4)',
                passed: result.success,
                result: result.success ? 'Position loaded' : result.error
            });
        } catch (error) {
            testResults.push({
                test: 'Position loading (after e4)',
                passed: false,
                error: error.message
            });
        }
        
        try {
            const game = new Chess();
            const moves = game.moves();
            testResults.push({
                test: 'Move generation from start',
                passed: moves.length === 20,
                result: moves.length + ' moves (expected 20)'
            });
        } catch (error) {
            testResults.push({
                test: 'Move generation from start',
                passed: false,
                error: error.message
            });
        }
        
        try {
            const game = new Chess();
            const moveResult = game.move('e4');
            const expectedFEN = 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1';
            testResults.push({
                test: 'Move application (e4)',
                passed: game.fen() === expectedFEN && moveResult !== null,
                result: game.fen()
            });
        } catch (error) {
            testResults.push({
                test: 'Move application (e4)',
                passed: false,
                error: error.message
            });
        }
        
        try {
            const aiAvailable = typeof chessAI !== 'undefined' && typeof chessAI.selectMove === 'function';
            testResults.push({
                test: 'AI availability',
                passed: aiAvailable,
                result: aiAvailable ? 'AI is available' : 'AI not found'
            });
        } catch (error) {
            testResults.push({
                test: 'AI availability',
                passed: false,
                error: error.message
            });
        }
        
        try {
            const game = new Chess();
            const gameOverStart = isGameOver(game);
            
            const testGame = new Chess();
            testGame.move('f3');
            testGame.move('e5');
            testGame.move('g4');
            testGame.move('Qh4');
            
            const gameOverMate = isGameOver(testGame);
            
            testResults.push({
                test: 'Game over detection',
                passed: !gameOverStart && gameOverMate,
                result: 'Start: ' + gameOverStart + ', Checkmate: ' + gameOverMate
            });
        } catch (error) {
            testResults.push({
                test: 'Game over detection',
                passed: false,
                error: error.message
            });
        }
        
        try {
            const testFEN = 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1';
            const gameResult = createChessGame(testFEN);
            
            if (gameResult.success) {
                const aiMove = await chessAI.selectMove(gameResult.game, 2);
                const moveApplied = gameResult.game.move(aiMove);
                
                testResults.push({
                    test: 'AI move generation and application',
                    passed: aiMove !== null && moveApplied !== null,
                    result: 'AI move: ' + aiMove + ', Applied: ' + (moveApplied ? moveApplied.san : 'failed')
                });
            } else {
                testResults.push({
                    test: 'AI move generation and application',
                    passed: false,
                    error: 'Could not load test position'
                });
            }
        } catch (error) {
            testResults.push({
                test: 'AI move generation and application',
                passed: false,
                error: error.message
            });
        }
        
        console.log('=== UNIT TESTS COMPLETE ===');
        
        const passedTests = testResults.filter(t => t.passed).length;
        const totalTests = testResults.length;
        
        res.json({
            success: true,
            summary: passedTests + '/' + totalTests + ' tests passed',
            passRate: ((passedTests / totalTests) * 100).toFixed(1) + '%',
            allPassed: passedTests === totalTests,
            tests: testResults,
            timestamp: new Date().toISOString()
        });
        
    } catch (error) {
        console.error('Unit test error:', error);
        res.status(500).json({
            success: false,
            error: 'Unit test execution failed: ' + error.message,
            tests: testResults
        });
    }
});

process.on('SIGINT', async () => {
    console.log('\nReceived SIGINT. Saving all data before shutdown...');
    try {
        await chessAI.saveAllData();
        console.log('Data saved successfully. Shutting down.');
        process.exit(0);
    } catch (error) {
        console.error('Error saving data during shutdown:', error);
        process.exit(1);
    }
});

process.on('SIGTERM', async () => {
    console.log('\nReceived SIGTERM. Saving all data before shutdown...');
    try {
        await chessAI.saveAllData();
        console.log('Data saved successfully. Shutting down.');
        process.exit(0);
    } catch (error) {
        console.error('Error saving data during shutdown:', error);
        process.exit(1);
    }
});

app.listen(port, () => {
    console.log('RL Chess AI Server running at http://localhost:' + port);
    console.log('Current AI Stats:', chessAI.getStats());
    console.log('Data persistence enabled - training data will be saved and loaded automatically');
});