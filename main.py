import tkinter as tk
from tkinter import simpledialog, messagebox
import chess
import chess.svg
import pandas as pd
import os
import io
from PIL import Image, ImageTk
from wand.image import Image as WandImage

# Define the path for your dataset
dataset_path = 'chess_dataset.csv'

class ChessGUI:
    def __init__(self, master):
        self.master = master
        self.board = chess.Board()
        self.moves_data = []
        self.from_square = None  # Variable to store the starting square of the move

        # Set up the frame for the chess board
        self.plot_frame = tk.Frame(master)
        self.plot_frame.pack()

        self.update_board()

    def make_move(self, from_square, to_square):
        try:
            move = chess.Move(from_square=from_square, to_square=to_square)
            if move in self.board.legal_moves:
                self.board.push(move)
                self.update_board()

                if self.board.is_check():
                    messagebox.showinfo("Check", "Check!")

                # Record the move
                self.moves_data.append({
                    'fen_before_move': self.board.fen(),
                    'move': move.uci(),
                    'label': ''  # Label will be assigned later
                })

            else:
                messagebox.showerror("Invalid Move", "The move was invalid. Please try again.")

            if self.board.is_game_over(claim_draw=True):
                self.game_over()

        except ValueError:
            messagebox.showerror("Invalid Move", "The move was invalid. Please try again.")

    def update_board(self):
        # Clear the previous plot
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        # Convert SVG chess board to a PNG image using Wand
        svg_data = chess.svg.board(self.board).encode('utf-8')
        with WandImage(blob=svg_data, format='svg') as image:
            png_image = image.make_blob('png')

        # Use Pillow to open the PNG image
        image = Image.open(io.BytesIO(png_image))

        # Convert the image to a format Tkinter Canvas can handle
        photo = ImageTk.PhotoImage(image)

        # Create a label and place the image on it, then bind mouse events
        self.label = tk.Label(self.plot_frame, image=photo)
        self.label.image = photo  # Keep a reference to avoid garbage collection
        self.label.pack()
        self.label.bind("<Button-1>", self.on_click)
        self.label.bind("<B1-Motion>", self.on_drag)  # Bind drag event
        self.label.bind("<ButtonRelease-1>", self.on_release)

    def on_click(self, event):
        x = event.x
        y = event.y
        self.from_square = self.pixel_to_square(x, y)  # Store the starting square

    def on_drag(self, event):
        # This function will be called when the mouse is dragged
        pass  # We don't need to do anything here, but the function must exist

    def on_release(self, event):
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
        if os.path.exists(dataset_path):
            df.to_csv(dataset_path, mode='a', header=False, index=False)
        else:
            df.to_csv(dataset_path, index=False)
        messagebox.showinfo("Data Saved", "Game data saved to the dataset.")

def main():
    root = tk.Tk()
    gui = ChessGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
