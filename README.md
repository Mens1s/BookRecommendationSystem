
# Book Recommendation System

This project implements a book recommendation system based on collaborative filtering. It uses similarity measures to recommend books to users based on their preferences or similar book titles.

## Features
- **Book Recommendations:** Provides a list of recommended books based on a selected book.
- **Fuzzy Matching:** Handles misspelled or incorrect book titles by suggesting the closest matching book.
- **Similarity Calculation:** Calculates book similarity using cosine similarity or another matrix-based similarity measure.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/book-recommendation-system.git
   ```
2. Navigate to the project directory:
   ```bash
   cd book-recommendation-system
   ```
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Import the recommendation function:
   ```python
   from recommendation_system import recommend
   ```
2. Run the recommendation for a book:
   ```python
   recommend("Harry Potter and the Chamber of Secrets (Book 2)")
   ```
3. If the selected book is not found, the system will attempt to find the most similar book and recommend alternatives based on that.

## File Structure

- `datasets/` - Folder containing CSV files for Books, Ratings, and Users data.


## Requirements
- pandas
- numpy
- seaborn
- scikit-learn

You can install all the dependencies by running:
```bash
pip install ***
```

## Example Output

When you run the recommendation system, you will get recommendations similar to the following:

```
Recommendations for the book Harry Potter and the Chamber of Secrets (Book 2):
-----
1. Harry Potter and the Prisoner of Azkaban (Book 3)
2. Harry Potter and the Goblet of Fire (Book 4)
3. Harry Potter and the Sorcerer's Stone (Harry Potter (Paperback))
4. Harry Potter and the Sorcerer's Stone (Book 1)
5. Harry Potter and the Order of the Phoenix (Book 5)
6. Charlotte's Web (Trophy Newbery)
7. The Fellowship of the Ring (The Lord of the Rings, Part 1)
8. The Shelters of Stone (Earth's Children Series, No 5)
9. The Eye of the World (The Wheel of Time, Book 1)
10. Stiff: The Curious Lives of Human Cadavers
```

If a book is not found:
```
Book "Harry Potter and the Chamber of Secrets" not found. Finding the most similar book...
No similar books found.
```

## Contributing

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
