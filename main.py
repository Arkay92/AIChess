from tkinter import messagebox
from PIL import Image, ImageTk, ImageDraw
from wand.image import Image as WandImage
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torchvision import transforms
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import chess, os, io, random, queue, threading, torch, time
import chess.svg
import pandas as pd
import tkinter as tk
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import sklearn.metrics as metrics

# Constants
DATASET_PATH = 'chess_dataset.csv'
MODEL_FILE_PATH = 'chess_cnn_model.pth'
IMAGE_SIZE = (128, 128)
MAX_ENCODED_MOVE = 4096  # Define this constant at the top level
MIN_MOVES_FOR_TRAINING = 100

# Updated transformation pipeline to ensure 3-channel input
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for 3 channels
])

# Chess Dataset with Preprocessing and Augmentation
class ChessDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def augment_image(self, image):
        # Randomly choose to apply transformations
        if random.choice([True, False]):
            image = image.transpose(Image.FLIP_LEFT_RIGHT)  # Flip image
        
        if random.choice([True, False]):
            image = image.rotate(90)  # Rotate image

        return image

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        fen = self.data_frame.iloc[idx, 0]
        move_str = self.data_frame.iloc[idx, 1]
        move_encoded = self.encode_move(move_str)
        move_tensor = torch.tensor(move_encoded, dtype=torch.long)

        image = self.fen_to_image_enhanced(fen)
        if self.transform:
            image = self.transform(image)

        return image, move_tensor
    
    def encode_move(self, move):
        # Example: Adding simple logic to include promotion in encoding
        # This is a basic example; consider expanding further for all move types
        from_square = (ord(move[0]) - ord('a')) + 8 * (int(move[1]) - 1)
        to_square = (ord(move[2]) - ord('a')) + 8 * (int(move[3]) - 1)
        encoded_move = from_square * 64 + to_square

        if len(move) == 5:  # Promotion move
            # Simple encoding for promotion piece. 'q' -> 1, 'r' -> 2, 'b' -> 3, 'n' -> 4
            promotion_piece = {'q': 1, 'r': 2, 'b': 3, 'n': 4}.get(move[4], 0)
            encoded_move = encoded_move * 10 + promotion_piece

        # Ensure the encoded move is within the range
        encoded_move = min(encoded_move, MAX_ENCODED_MOVE - 1)

        return encoded_move

    def fen_to_image_enhanced(self, fen):
        board = chess.Board(fen)
        # Initialize a 8x8x3 array to represent the board in 3 channels
        array = np.zeros((8, 8, 3), dtype=np.float32)
        
        # Mapping of pieces to channels
        piece_to_channel = {
            'P': 0,  # Pawns in channel 0
            'N': 1, 'B': 1,  # Knights and Bishops in channel 1
            'R': 2, 'Q': 2, 'K': 2,  # Rooks, Queens, and Kings in channel 2
            'p': 0,  # Black pawns in channel 0 (negative values)
            'n': 1, 'b': 1,  # Black knights and bishops in channel 1 (negative values)
            'r': 2, 'q': 2, 'k': 2   # Black rooks, queens, and kings in channel 2 (negative values)
        }

        for square, piece in board.piece_map().items():
            rank, file = divmod(square, 8)
            channel = piece_to_channel[str(piece)]
            array[rank, file, channel] = 1 if piece.color else -1  # Positive for white, negative for black

        # Convert the array to an image
        image = Image.fromarray(np.uint8((array + 1) / 2 * 255), 'RGB')  # Normalize and convert to RGB image
        return image

    def fen_to_image(self, fen, square_size=60, augment=False):
        board = chess.Board(fen)
        img_size = square_size * 8
        image = Image.new("RGB", (img_size, img_size), "white")
        draw = ImageDraw.Draw(image)
        
        # Draw the squares
        for i in range(8):
            for j in range(8):
                color = "white" if (i + j) % 2 == 0 else "gray"
                self.draw_square(draw, color, j, i, square_size)
        
        # Draw the pieces
        piece_symbols = {"P": "♙", "R": "♖", "N": "♘", "B": "♗", "Q": "♕", "K": "♔", "p": "♟", "r": "♜", "n": "♞", "b": "♝", "q": "♛", "k": "♚"}
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                symbol = piece_symbols[str(piece)]
                x = chess.square_file(square)
                y = 7 - chess.square_rank(square)
                draw.text((x * square_size + square_size // 4, y * square_size), symbol, fill="black")
        
        if augment and random.choice([True, False]):
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        return image

    def draw_square(self, draw, color, x, y, square_size):
        draw.rectangle([x * square_size, y * square_size, (x + 1) * square_size, (y + 1) * square_size], fill=color)

# Updated ChessCNN Model
class ChessCNN(nn.Module):
    def __init__(self):
        super(ChessCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(0.25)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.dropout2 = nn.Dropout(0.25)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # Adjusted the input features of self.fc1 to 262144
        self.fc1 = nn.Linear(262144, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, MAX_ENCODED_MOVE)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)  # Ensure flattening maintains the batch size
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ChessGUI:
    def __init__(self, master, transform=None):  # Add 'transform' parameter
        self.master = master
        self.transform = transform  # Store it as an instance attribute
        self.board = chess.Board()
        self.moves_data = []
        self.plot_frame = tk.Frame(master)
        self.plot_frame.pack()
        self.training_button = tk.Button(master, text="Start Training", command=self.toggle_training_mode)
        self.training_button.pack()
        self.check_dataset_availability()
        self.image_queue = queue.Queue()
        self.move_queue = queue.Queue()
        self.is_training_mode = False  # Ensure this is initialized before update_board is called
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChessCNN().to(self.device)
        self.load_model()
        self.update_board()  # Now safe to call as is_training_mode is initialized
        self.games_played = 0  # Initialize a counter to track the number of games played in automated mode
        self.game_in_progress = False

    def toggle_training_mode(self):
        if self.is_training_mode:
            self.exit_training_mode()
        else:
            self.enter_training_mode()

    def load_model(self):
        if os.path.exists(MODEL_FILE_PATH):
            self.model.load_state_dict(torch.load(MODEL_FILE_PATH, map_location=self.device))

    def make_move(self, from_square, to_square):
        try:
            # Check if the move is a pawn promotion (pawn reaches the last rank)
            promotion = None
            moving_piece = self.board.piece_at(from_square)
            if moving_piece and moving_piece.piece_type == chess.PAWN:
                if to_square in chess.SquareSet(chess.BB_RANK_1 | chess.BB_RANK_8):
                    # Promote to a queen by default
                    promotion = chess.QUEEN

            move = chess.Move(from_square=from_square, to_square=to_square, promotion=promotion)

            if move in self.board.legal_moves:
                self.board.push(move)
                self.update_board()

                if self.board.is_check():
                    messagebox.showinfo("Check", "Check!")

                self.moves_data.append({
                    'fen_before_move': self.board.fen(),
                    'move': move.uci(),
                    'label': ''
                })

                # Only make an AI move if it's AI's turn and not in training mode
                if not self.is_training_mode and not self.board.turn:
                    self.master.after(1000, self.make_ai_move)

            else:
                messagebox.showerror("Invalid Move", "The move was invalid. Please try again.")

            if self.board.is_game_over(claim_draw=True):
                self.game_over()

        except ValueError:
            messagebox.showerror("Invalid Move", "The move was invalid. Please try again.")

    def make_ai_move(self):
        # Check if the game is over or if it's not the AI's turn (AI is black).
        if self.board.is_game_over(claim_draw=True) or self.board.turn:
            return  # Do nothing if the game is over or if it's not the AI's turn.

        # Generate an AI move.
        board_image = self.capture_board_state_as_image()
        ai_move = self.get_nn_move(board_image)

        # Ensure the generated move is legal and for the AI's own pieces.
        if ai_move in self.board.legal_moves and self.board.color_at(ai_move.from_square) == chess.BLACK:
            self.board.push(ai_move)  # Make the move on the board.
            self.update_board()  # Update the board to reflect the move.
        else:
            # Log an error or handle the case where an invalid or illegal move was generated.
            print("AI generated an invalid move or tried to move an opponent")

    def cnn_predict_move(self):
        while True:
            board_image = self.capture_board_state_as_image()
            move = self.get_nn_move(board_image)
            # Validate the move before putting it in the queue
            if move in self.board.legal_moves:
                self.move_queue.put(move)
            else:
                print("Invalid move predicted. Ignoring the move.")

    def get_nn_move(self, board_image):
        image_tensor = self.pil_to_tensor(board_image).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(image_tensor)
        # Translate the prediction to a chess move
        move = self.translate_prediction_to_move(prediction)
        return move

    def translate_prediction_to_move(self, prediction):
        # Get all legal moves for the AI (assuming AI is black)
        legal_moves = [move for move in self.board.legal_moves if self.board.color_at(move.from_square) == chess.BLACK]

        # If there are no legal moves, return None
        if not legal_moves:
            return None

        # Convert the neural network's prediction to an index
        move_index = torch.argmax(prediction).item()

        # Map the prediction index to a legal move
        # Note: This is a simplified approach; you may need a more sophisticated method
        # to match predictions to legal moves, especially if the neural network's output
        # does not directly correspond to specific moves.
        selected_move = legal_moves[move_index % len(legal_moves)]  # Use modulo to ensure the index is within bounds

        return selected_move

    def pil_to_tensor(self, image):
        # Ensure image is in RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_tensor = self.transform(image)
        return image_tensor

    def start_training_thread(self):
        # Ensure there's no ongoing training thread
        if hasattr(self, 'training_thread') and self.training_thread.is_alive():
            print("Training already in progress")
            return

        # Load data
        train_loader, test_loader = load_data(DATASET_PATH)
        
        # Define the number of epochs for training
        num_epochs = 10
        
        # Start the training thread and pass the required arguments
        self.training_thread = threading.Thread(target=self.train_model_background,
                                                args=(self.model, self.device, train_loader, test_loader, num_epochs))
        self.training_thread.daemon = True
        self.training_thread.start()

    
    def train_model_background(self, model, device, train_loader, test_loader, num_epochs):
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        for epoch in range(num_epochs):
            model.train()
            for images, moves in train_loader:
                images = images.to(device)
                moves = moves.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, moves)  # 'moves' should be a tensor of integers representing the move indexes
                loss.backward()
                optimizer.step()

            # Validation Phase
            evaluate_model(model, test_loader, device, epoch)  # Pass epoch to evaluate_model function

        # After training, save the model
        torch.save(model.state_dict(), MODEL_FILE_PATH)
        print(f'Model saved to {MODEL_FILE_PATH}')

    # Modify the enter_training_mode function
    def enter_training_mode(self):
        print("Entering training mode")
        self.is_training_mode = True
        self.training_button.config(text="Stop Training")
        self.board.reset()
        self.is_training_mode = True

        # Prepare data loaders before starting the training thread
        train_loader, test_loader = self.prepare_data_loaders()
        self.start_training_thread(train_loader, test_loader)

        self.game_in_progress = True
        self.automated_move()  # Start the first move without waiting

    def prepare_data_loaders(self):
        # Assuming you have a function to load data
        return load_data(DATASET_PATH)

    def automated_game(self):
        while self.is_training_mode and not self.board.is_game_over(claim_draw=True):
            # Get the current board state as an image
            board_image = self.capture_board_state_as_image()

            # Get the neural network's predicted move for the current player
            nn_move = self.get_nn_move(board_image)

            # If the predicted move is legal, make the move on the board
            if nn_move in self.board.legal_moves:
                self.board.push(nn_move)
            else:
                # If the predicted move is not legal, choose a random legal move
                # This is a fallback strategy and might be refined
                self.board.push(random.choice(list(self.board.legal_moves)))

            # Update the GUI to reflect the new board state
            self.master.after(0, self.update_board)

            # Add a small delay to make the game progress visible and to prevent overloading the CPU
            time.sleep(0.5)

        # Check for game over and handle accordingly
        if self.board.is_game_over(claim_draw=True):
            self.game_over()  # Call the modified game_over method that handles automated training mode

    def display_next_move_in_training(self):
        if self.current_move_index < len(self.training_data):
            move_data = self.training_data.iloc[self.current_move_index]
            self.board.set_fen(move_data['fen_before_move'])
            self.update_board()
            self.current_move_index += 1
        else:
            messagebox.showinfo("Training Mode", "No more moves to display.")
            self.current_move_index = 0  # Reset for the next training session

    def save_board_image(self, image_data):
        # Specify the path where you want to save the image
        # Ensure the directory exists or choose a directory that does
        image_path = "images/board_image.png"  # Saves in the same directory as the script
        
        # Open a file in write-binary ('wb') mode
        with open(image_path, 'wb') as f:
            f.write(image_data)
        
        # Optionally, print a message or log the saving
        print(f"Board image saved to {image_path}")

    def exit_training_mode(self):
        self.is_training_mode = False
        self.training_button.config(text="Start Training")
        self.game_in_progress = False  # Signal to stop any automated moves
        self.board.reset()  # Reset the board to its initial state
        self.update_board()  # Update the board to reflect the reset state

    def update_board(self):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        svg_data = chess.svg.board(self.board).encode('utf-8')
        with WandImage(blob=svg_data, format='svg') as image:
            png_image = image.make_blob('png')

        image = Image.open(io.BytesIO(png_image))
        photo = ImageTk.PhotoImage(image)

        self.save_board_image(png_image)

        self.label = tk.Label(self.plot_frame, image=photo)
        self.label.image = photo
        self.label.pack()

        # Bind or unbind events based on the mode
        if self.is_training_mode:
            self.label.unbind("<Button-1>")
            self.label.unbind("<B1-Motion>")
            self.label.unbind("<ButtonRelease-1>")
        else:
            self.label.bind("<Button-1>", self.on_click)
            self.label.bind("<B1-Motion>", self.on_drag)
            self.label.bind("<ButtonRelease-1>", self.on_release)

            # If it's AI's turn (not in training mode), make an AI move
            if not self.board.turn:  # Assuming the AI plays as Black
                self.master.after(1000, self.make_ai_move)

    def capture_board_state_as_image(self):
        # Convert the current board state to an image
        svg_data = chess.svg.board(self.board).encode('utf-8')
        with WandImage(blob=svg_data, format='svg') as image:
            png_image = image.make_blob('png')

        # Ensure the image is opened as a PIL Image
        board_image = Image.open(io.BytesIO(png_image))

        return board_image

    def on_click(self, event):
        x = event.x
        y = event.y
        self.from_square = self.pixel_to_square(x, y)  # Store the starting square
        self.label.config(cursor="hand2")  # Change cursor to indicate dragging

    def on_drag(self, event):
        # This function will be called when the mouse is dragged
        pass  # We don't need to do anything here, but the function must exist

    def on_release(self, event):
        self.label.config(cursor="")  # Reset cursor to default
        x = event.x
        y = event.y
        to_square = self.pixel_to_square(x, y)
        self.make_move(self.from_square, to_square)  # Pass the starting square to make_move

    def pixel_to_square(self, x, y):
        # Assuming 8x8 grid and each square is 50x50 pixels
        row = 7 - (y // 50)
        col = x // 50
        return chess.square(col, row)
    
    def automated_move(self):
        if self.is_training_mode and not self.board.is_game_over(claim_draw=True) and self.game_in_progress:
            # Get the current board state as an image
            board_image = self.capture_board_state_as_image()

            # Get the neural network's predicted move for the current player
            nn_move = self.get_nn_move(board_image)

            # Make the move if it's legal
            if nn_move in self.board.legal_moves:
                self.board.push(nn_move)
            else:
                # If the predicted move is not legal, choose a random legal move
                self.board.push(random.choice(list(self.board.legal_moves)))

            self.update_board()

            # Schedule the next move
            self.master.after(500, self.automated_move)  # 500 ms delay between moves
        else:
            self.game_over()  # Handle the game over scenario

    def restart_game(self):
        # Your existing restart_game logic...
        if self.games_played >= 10:
            self.is_training_mode = False
            self.game_in_progress = False  # Stop the game
            self.games_played = 0  # Reset the games played counter
            print("Completed 10 games in automated training mode. Exiting training mode.")
        else:
            # Start a new game immediately without waiting
            self.board.reset()
            self.update_board()
            self.moves_data.clear()  # Clear the moves data for the new game
            self.games_played += 1  # Increment the games played counter
            self.automated_move()  # Start the next game immediately

    # Modify the game_over method
    def game_over(self):
        if not self.is_training_mode:
            outcome = self.check_end_condition()

            # Label all moves based on game outcome
            for move_data in self.moves_data:
                move_data['label'] = outcome

            # Save the game data to the dataset
            self.save_dataset()
            self.restart_game()
        else:
            # In training mode, just restart the game without showing any alerts
            self.restart_game()
            pass

    def check_end_condition(self):
        if self.board.is_checkmate():
            return 'win' if self.board.turn == chess.BLACK else 'loss'
        elif self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.can_claim_fifty_moves() or self.board.is_fivefold_repetition() or self.board.is_seventyfive_moves():
            return 'draw'
        else:
            return 'unknown'

    def check_dataset_availability(self):
        if not os.path.exists(DATASET_PATH) or pd.read_csv(DATASET_PATH).shape[0] < MIN_MOVES_FOR_TRAINING:
            print("No sufficient dataset found. Starting automated data generation...")
            self.automated_data_generation()
        else:
            self.training_button.config(state=tk.NORMAL)
            
    def automated_data_generation(self):
        # Create a new thread for automated data generation
        threading.Thread(target=self.generate_data, daemon=True).start()
    
    def generate_data(self):
        move_data = []
        while len(move_data) < MIN_MOVES_FOR_TRAINING:
            board = chess.Board()
            while not board.is_game_over(claim_draw=True):
                move = random.choice(list(board.legal_moves))
                board.push(move)
                move_data.append({
                    'fen_before_move': board.fen(),
                    'move': move.uci()
                })
            print(f"Generated {len(move_data)} moves")

        # Save the generated data
        df = pd.DataFrame(move_data)
        df.to_csv(DATASET_PATH, index=False)
        print(f"Saved generated data to {DATASET_PATH}")

        # Start training in the background
        self.start_background_training()

    def start_background_training(self):
        print("Starting training in the background...")
        # Your training setup code here
        train_loader, test_loader = load_data(DATASET_PATH)
        self.start_training_thread(train_loader, test_loader)

    def start_training_thread(self, train_loader, test_loader):
        # Define the number of epochs for training
        num_epochs = 10
        
        # Start the training thread and pass the required arguments
        self.training_thread = threading.Thread(target=self.train_model_background,
                                                args=(self.model, self.device, train_loader, test_loader, num_epochs))
        self.training_thread.daemon = True
        self.training_thread.start()
    
    def save_dataset(self):
        if self.moves_data:
            df = pd.DataFrame(self.moves_data)
            # Append data to the CSV file, change 'a' to 'w' if you want to overwrite each time
            df.to_csv(DATASET_PATH, mode='a', header=not os.path.exists(DATASET_PATH), index=False)
            print("Game data saved to dataset.")

def evaluate_model(model, test_loader, device, epoch):
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted_top1 = torch.max(outputs, 1)
            _, predicted_top5 = outputs.topk(5, dim=1)
            
            total += labels.size(0)
            correct_top1 += (predicted_top1 == labels).sum().item()
            correct_top5 += sum([labels[i] in predicted_top5[i] for i in range(len(labels))])
    
    print(f'Epoch {epoch + 1}, Top-1 Accuracy: {100 * correct_top1 / total:.2f}%, Top-5 Accuracy: {100 * correct_top5 / total:.2f}%')

# Data Preparation
def load_data(csv_file):
    dataset = ChessDataset(csv_file=csv_file, transform=transform)
    train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64)
    return train_loader, test_loader

# Updated Training Function
def train_model(model, device, train_loader, test_loader, optimizer, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, moves in train_loader:
            images, moves = images.to(device), moves.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, moves)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

        # Validation phase
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, moves in test_loader:
                images, moves = images.to(device), moves.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += moves.size(0)
                correct += (predicted == moves).sum().item()

            print(f'Validation Accuracy: {100 * correct / total:.2f}%')

# Main function
def main():
    # Initialize Tkinter GUI
    root = tk.Tk()
    gui = ChessGUI(root, transform=transform)
    root.mainloop()

    # Prepare data loaders
    train_loader, test_loader = load_data(DATASET_PATH)

    # Initialize model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, device, train_loader, test_loader, optimizer)

    # Save the trained model
    torch.save(model.state_dict(), MODEL_FILE_PATH)
    print(f'Model saved to {MODEL_FILE_PATH}')

if __name__ == "__main__":
    main()