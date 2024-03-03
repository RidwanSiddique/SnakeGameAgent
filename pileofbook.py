# Read input values
starting_height, stable_height, partition = map(int, input().split())

# Initialize a variable to keep track of the number of piles
num_piles = 0

# Loop until the starting height is greater than the stable height
while starting_height > stable_height:
    # Calculate the number of piles after attempting to split into 'partition'
    piles = (starting_height + partition - 1) // partition

    # Calculate the height of each pile after splitting.
    pile_height = (starting_height + piles - 1) // piles

    # Check if the pile height exceeds the stable height
    if pile_height > stable_height:
        # Update the starting height and increment the number of piles
        starting_height = pile_height
        num_piles += piles
    else:
        # If the pile height is not greater than the stable height, break the loop
        break

# Increment the number of piles by 1 if there are remaining books
if starting_height > stable_height:
    num_piles += 1

# Print the result
print(num_piles)
