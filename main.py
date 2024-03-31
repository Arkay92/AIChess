from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
from wand.image import Image as WandImage
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torchvision import transforms
from sklearn.metrics import precision_score, recall_score, f1_score
import chess, os, io, random, queue, threading, torch, time
import chess.svg
import pandas as pd
import tkinter as tk
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

# Constants
DATASET_PATH = 'chess_dataset.csv'
MODEL_FILE_PATH = 'chess_cnn_model.pth'
IMAGE_SIZE = (128, 128)
MAX_ENCODED_MOVE = 4096  # Define this constant at the top level
MIN_MOVES_FOR_TRAINING = 100

# Transformation pipeline for image preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Chess Dataset with Preprocessing and Augmentation
class ChessDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        fen = self.data_frame.iloc[idx, 0]
        move_str = self.data_frame.iloc[idx, 1]
        move_encoded = self.encode_move(move_str)
        move_tensor = torch.tensor(move_encoded, dtype=torch.long)

        image = self.fen_to_image(fen)
        if self.transform:
            image = self.transform(image)

        return image, move_tensor
    
    def encode_move(self, move):
        # Assuming each square on the chessboard is assigned a unique number from 0-63
        # e.g., "a1" -> 0, "b1" -> 1, ..., "h8" -> 63
        from_square = (ord(move[0]) - ord('a')) + 8 * (int(move[1]) - 1)
        to_square = (ord(move[2]) - ord('a')) + 8 * (int(move[3]) - 1)
        return from_square * 64 + to_square

    def fen_to_image(self, fen, square_size=60):
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

        return image

    def draw_square(self, draw, color, x, y, square_size):
        draw.rectangle([x * square_size, y * square_size, (x + 1) * square_size, (y + 1) * square_size], fill=color)

# Model Definition with Softmax Activation
class ChessCNN(nn.Module):
    def __init__(self):
        super(ChessCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(32 * IMAGE_SIZE[0]//4 * IMAGE_SIZE[1]//4, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, MAX_ENCODED_MOVE)  # Now MAX_ENCODED_MOVE is recognized

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * IMAGE_SIZE[0]//4 * IMAGE_SIZE[1]//4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No softmax here, since nn.CrossEntropyLoss will be used
        return x

class ChessGUI:
    def __init__(self, master, transform=None):  # Add 'transform' parameter
        self.master = master
        self.transform = transform  # Store it as an instance attribute
        self.board = chess.Board()
        self.moves_data = []
        self.plot_frame = tk.Frame(master)
        self.plot_frame.pack()
        self.training_button = tk.Button(master, text="Training Mode", command=self.enter_training_mode)
        self.training_button.pack()
        self.check_dataset_availability()
        self.image_queue = queue.Queue()
        self.move_queue = queue.Queue()
        self.is_training_mode = False  # Ensure this is initialized before update_board is called
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChessCNN().to(self.device)
        self.load_model()
        self.update_board()  # Now safe to call as is_training_mode is initialized
        self.automated_game_thread = None  # Add this line to initialize the automated game thread

    def load_model(self):
        if os.path.exists(MODEL_FILE_PATH):
            self.model.load_state_dict(torch.load(MODEL_FILE_PATH, map_location=self.device))

    def make_move(self, from_square, to_square):
        try:
            move = chess.Move(from_square=from_square, to_square=to_square)
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
                if not self.is_training_mode and not self.board.turn:  # Adjusted logic here
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
            print("AI generated an invalid move or tried to move an opponent's piece.")

    def start_cnn_thread(self):
        threading.Thread(target=self.cnn_predict_move, daemon=True).start()

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

    def square_to_uci(self, square):
        # Convert a square index to UCI format (e.g., index 0 -> 'a1')
        row = square % 8
        col = square // 8
        return f"{chr(ord('a') + col)}{row + 1}"

    def pil_to_tensor(self, image):
        # Convert a PIL Image to a PyTorch tensor using the transformation pipeline
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

        for epoch in range(num_epochs):
            model.train()
            for images, moves in train_loader:  # Directly unpack the batch
                images = images.to(device)  # Move images to the correct device
                moves = moves.to(device)  # Move moves to the correct device

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, moves)
                loss.backward()
                optimizer.step()

            # Validation Phase
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for images, moves in test_loader:
                    images = images.to(device)
                    moves = moves.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += moves.size(0)
                    correct += (predicted == moves).sum().item()

            print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}, Validation Accuracy: {100 * correct / total:.2f}%')

        # After training, save the model
        torch.save(model.state_dict(), MODEL_FILE_PATH)
        print(f'Model saved to {MODEL_FILE_PATH}')

    # Modify the enter_training_mode function
    def enter_training_mode(self):
        print("Entering training mode")
        self.board.reset()
        self.is_training_mode = True
        self.start_training_thread()  # Start training in a separate thread
        if self.automated_game_thread is None or not self.automated_game_thread.is_alive():
            self.automated_game_thread = threading.Thread(target=self.automated_game, daemon=True)
            self.automated_game_thread.start()

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
            self.game_over()

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

    def game_over(self):
        outcome = self.check_end_condition()

        # Label all moves based on game outcome
        for move_data in self.moves_data:
            move_data['label'] = outcome

        # Save the game data to the dataset
        self.save_dataset()

        messagebox.showinfo("Game Over", "The game is over. Outcome: " + outcome)

    def check_end_condition(self):
        if self.board.is_checkmate():
            return 'win' if self.board.turn == chess.BLACK else 'loss'
        elif self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.can_claim_fifty_moves() or self.board.is_fivefold_repetition() or self.board.is_seventyfive_moves():
            return 'draw'
        else:
            return 'unknown'

    def save_dataset(self):
        df = pd.DataFrame(self.moves_data)
        if os.path.exists(DATASET_PATH):
            df.to_csv(DATASET_PATH, mode='a', header=False, index=False)
        else:
            df.to_csv(DATASET_PATH, index=False)

        self.check_dataset_availability()  # Check if training mode should be enabled

    def check_dataset_availability(self):
        if os.path.exists(DATASET_PATH):
            # Read the dataset to check its length
            df = pd.read_csv(DATASET_PATH)
            if len(df) >= MIN_MOVES_FOR_TRAINING:  # Check if there are enough moves for training
                self.training_button.config(state=tk.NORMAL)
            else:
                self.training_button.config(state=tk.DISABLED)
                messagebox.showinfo("Training Mode", f"Training requires at least {MIN_MOVES_FOR_TRAINING} moves. Current dataset contains only {len(df)} moves.")
        else:
            self.training_button.config(state=tk.DISABLED)
            messagebox.showinfo("Training Mode", "No dataset found. Please create a dataset for training.")

# Data Preparation
def load_data(csv_file):
    dataset = ChessDataset(csv_file=csv_file, transform=transform)
    train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64)
    return train_loader, test_loader

# Training and Validation
def train_model(model, device, train_loader, test_loader, optimizer, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        model.train()
        for images, moves in train_loader:
            images, moves = images.to(device), moves.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, moves)
            loss.backward()
            optimizer.step()

        # Validation Phase
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

            print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}, Validation Accuracy: {100 * correct / total:.2f}%')

def main():
    root = tk.Tk()
    gui = ChessGUI(root, transform=transform)
    root.mainloop()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_data(DATASET_PATH)

    model = ChessCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Pass both train_loader and test_loader to the train_model function
    train_model(model, device, train_loader, test_loader, optimizer)

    torch.save(model.state_dict(), MODEL_FILE_PATH)
    print(f'Model saved to {MODEL_FILE_PATH}')

if __name__ == "__main__":
    main()