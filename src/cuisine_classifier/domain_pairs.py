"""
Domain pairs for cuisine cosine-similarity fine-tuning.

Raw data  : my_domain_pairs
  Each tuple → (cuisine_domain, dish_name, description, ingredients, label)
  label "close" → dish belongs to that cuisine
  label "far"   → dish is from a different cuisine

  120 total pairs (20 per cuisine × 6 cuisines, 10 close + 10 far each).
  60 pairs use exact dish name / description / ingredients from the AllRecipes dataset.
  60 pairs use general dish terms with standard descriptions.

Ready-to-use: get_train_examples()
  Returns a list of (anchor, dish_text, label) tuples.
    anchor    : e.g. "american cuisine"
    dish_text : dish name + description + ingredients as one string
    label     : "close" or "far"

  To convert to InputExample for CosineSimilarityLoss:
    LABEL_MAP = {"close": 1.0, "far": 0.0}
    examples = [InputExample(texts=[a, d], label=LABEL_MAP[l])
                for a, d, l in get_train_examples()]
"""


def _dish_text(dish_name: str, description: str, ingredients: str) -> str:
    return (
        f"{dish_name}. "
        f"{description} "
        f"Ingredients: {ingredients}"
    )


def get_train_examples() -> list[tuple[str, str, str]]:
    """Return (anchor, dish_text, label) tuples. label is 'close' or 'far'."""
    return [
        (
            f"{cuisine} cuisine",
            _dish_text(dish_name, description, ingredients),
            label,
        )
        for cuisine, dish_name, description, ingredients, label in my_domain_pairs
    ]

my_domain_pairs = [
    # ============================================================
    # American — 10 dataset-backed pairs
    # ============================================================
    (
        "american",
        "Smash Burgers",
        "This smash burger recipe makes super juicy burgers with crispy edges. "
        "I prefer to cook these outdoors — they grill up very fast because of the high heat, "
        "so make sure you have everything ready to go!",
        "4 hamburger buns | 2 tablespoons butter, softened, or as needed | "
        "1 pound ground chuck beef (80% lean) | 4 (6-inch) squares parchment paper | "
        "salt to taste | 4 slices American cheese | burger toppings of choice",
        "close",
    ),
    (
        "american",
        "Basic Air Fryer Hot Dogs",
        "These air fryer hot dogs in toasted buns turn out perfectly crisp in minutes "
        "with the help of your air fryer. Top with ketchup, mustard, relish, chili, "
        "or condiments of choice.",
        "4 hot dog buns | 4 hot dogs",
        "close",
    ),
    (
        "american",
        "Chicago-Style Hot Dog",
        "This hot dog is a Windy City classic and a big favorite with sports fans! "
        "The frank must be all-beef, the bun must be poppy seed, and the ingredients "
        "must be piled onto the bun in the order specified. And whatever you do, "
        "don't spoil the splendor of this Chicago dog with ketchup!",
        "1 all-beef hot dog | 1 poppy seed hot dog bun | 1 tablespoon yellow mustard | "
        "1 tablespoon sweet green pickle relish | 1 tablespoon chopped onion | "
        "2 tomato wedges | 1 dill pickle spear | 2 sport peppers | 1 dash celery salt",
        "close",
    ),
    (
        "american",
        "Grandma's Buttermilk Cornbread",
        "Fluffy, moist skillet cornbread—just like Nana used to make! Serve with butter and honey.",
        "1 1/4 cups stone-ground cornmeal | 3/4 cup all-purpose flour | 1/4 cup white sugar | "
        "2 teaspoons baking powder | 1 teaspoon salt | 1/2 teaspoon baking soda | "
        "2 large eggs | 1/3 cup vegetable oil | 1 1/2 cups buttermilk | 2 tablespoons buttermilk",
        "close",
    ),
    (
        "american",
        "Buffalo Chicken Mac and Cheese",
        "This Buffalo chicken mac and cheese recipe was first introduced to me at my local "
        "ballpark, and I thought it was a great idea! Serve with celery sticks and beer.",
        "1 (16 ounce) package elbow macaroni | 1 rotisserie-roasted chicken | "
        "6 tablespoons butter | 6 tablespoons all-purpose flour | 3 cups milk | "
        "1 pinch ground black pepper | 2 cups shredded Cheddar cheese | "
        "2 cups shredded Monterey Jack cheese | 1/2 cup hot sauce | "
        "1/2 cup crumbled gorgonzola cheese",
        "close",
    ),
    (
        "american",
        "Chicken Biryani",
        "Chicken biryani is a delicious Pakistani/Indian rice dish that's typically reserved "
        "for special occasions such as weddings, parties, or holidays such as Ramadan. "
        "It has a lengthy preparation, but the work is definitely worth it.",
        "4 tablespoons vegetable oil | 4 small potatoes, peeled and halved | "
        "2 large onions, finely chopped | 2 cloves garlic, minced | "
        "1 tablespoon minced fresh ginger root | 2 medium tomatoes, peeled and chopped | "
        "1 teaspoon salt | 1 teaspoon ground cumin | 1/2 teaspoon chili powder | "
        "1/2 teaspoon ground turmeric | 2 tablespoons plain yogurt | "
        "1/2 teaspoon ground cardamom | 3 pounds boneless chicken pieces | "
        "1 pound basmati rice | 1 pinch powdered saffron | 4 cups chicken stock",
        "far",
    ),
    (
        "american",
        "Homemade Wonton Soup",
        "This is a recipe I have perfected on my own through the years. "
        "This recipe is one that my family members beg me to make all the time!",
        "1 bunch green onions | 6 fresh mushrooms, sliced | 1 pound ground pork | "
        "1 tablespoon sesame oil | 1 tablespoon soy sauce | 1 egg | "
        "1/4 cup dry bread crumbs | 1/4 teaspoon salt | "
        "1 (16 ounce) package wonton wrappers | 8 cups chicken broth | "
        "16 uncooked medium shrimp, peeled and deveined | "
        "1 medium head bok choy | 16 snow peas | 1 dash sesame oil",
        "far",
    ),
    (
        "american",
        "White Cheese Chicken Lasagna",
        "Chicken lasagna with spinach and a creamy white cheese sauce. "
        "Great for any kind of potluck. My kids love it!",
        "9 lasagna noodles | 1/2 cup butter | 1 onion, chopped | 1 clove garlic, minced | "
        "1/2 cup all-purpose flour | 2 cups chicken broth | 1 1/2 cups milk | 1 teaspoon salt | "
        "4 cups shredded mozzarella cheese | 1 cup grated Parmesan cheese | "
        "1 teaspoon dried basil | 1 teaspoon dried oregano | 2 cups ricotta cheese | "
        "2 cups cubed cooked chicken | 2 packages frozen chopped spinach",
        "far",
    ),
    (
        "american",
        "Barbacoa Tacos",
        "These barbacoa tacos are packed with smoky shredded beef that's perfectly tender. "
        "Spices like cumin complement the chiles, while oregano and bay leaves add an "
        "earthiness to this recipe.",
        "2 ripe plum tomatoes | 1 small white onion, quartered | 2 cloves garlic, peeled | "
        "4 chipotle peppers in adobo sauce | 3 teaspoons kosher salt | 1 1/2 teaspoons chili powder | "
        "1 teaspoon ground cumin | 1 (3 pound) beef chuck roast | 2 tablespoons olive oil | "
        "2 teaspoons dried oregano | 3 fresh bay leaves | 1 tablespoon lime juice | "
        "corn tortillas | 2 ripe avocados | 2 tablespoons chopped fresh cilantro",
        "far",
    ),
    (
        "american",
        "Paneer Tikka Masala",
        "This paneer tikka masala made with cubes of paneer cheese in a spicy creamy curry sauce "
        "is easy to make. This vegetarian dish goes well with naan or basmati rice.",
        "1/4 cup butter | 1 pound paneer, cut into 1/2-inch cubes | 2 medium onions, finely chopped | "
        "1 medium green bell pepper | 2 medium jalapeno peppers | 1 tablespoon ground cashews | "
        "1 teaspoon garlic paste | 1 teaspoon ginger paste | 1 teaspoon cayenne pepper | "
        "1 teaspoon ground cumin | 1 teaspoon garam masala | "
        "1 (16 ounce) can tomato sauce | 1 pint half-and-half | 1 teaspoon salt",
        "far",
    ),

    # American — 10 general pairs
    (
        "american",
        "burger",
        "A classic American sandwich featuring a grilled beef patty served in a soft bun "
        "with lettuce, tomato, onion, and condiments.",
        "ground beef | burger bun | lettuce | tomato | onion | pickles | ketchup | mustard | American cheese",
        "close",
    ),
    (
        "american",
        "hot dog",
        "An American staple: a pork or beef frankfurter served in a soft bun, "
        "topped with mustard, ketchup, and relish.",
        "hot dog sausage | hot dog bun | yellow mustard | ketchup | sweet relish | onion",
        "close",
    ),
    (
        "american",
        "barbecue ribs",
        "Slow-smoked pork or beef ribs glazed with a smoky, sweet barbecue sauce — "
        "a backyard cookout staple.",
        "pork baby back ribs | barbecue sauce | brown sugar | smoked paprika | "
        "garlic powder | onion powder | salt | black pepper | apple cider vinegar",
        "close",
    ),
    (
        "american",
        "mac and cheese",
        "Creamy baked macaroni pasta smothered in a rich cheddar cheese sauce, "
        "the ultimate American comfort food.",
        "elbow macaroni | cheddar cheese | butter | whole milk | all-purpose flour | "
        "salt | black pepper | mustard powder | breadcrumbs",
        "close",
    ),
    (
        "american",
        "apple pie",
        "A traditional American dessert with a flaky butter crust filled with "
        "cinnamon-spiced apples and baked until golden.",
        "all-purpose flour | unsalted butter | Granny Smith apples | white sugar | "
        "brown sugar | cinnamon | nutmeg | lemon juice | salt | ice water",
        "close",
    ),
    (
        "american",
        "sushi",
        "A Japanese dish of vinegared rice paired with fresh fish, seafood, or vegetables, "
        "served with soy sauce, wasabi, and pickled ginger.",
        "sushi rice | nori | salmon | tuna | cucumber | avocado | soy sauce | "
        "wasabi | pickled ginger | rice vinegar | sugar | salt",
        "far",
    ),
    (
        "american",
        "biryani",
        "A fragrant South Asian layered rice dish cooked with aromatic spices, "
        "marinated meat, and caramelized onions.",
        "basmati rice | chicken | onion | plain yogurt | garam masala | turmeric | "
        "cumin | coriander | saffron | ghee | fresh mint | bay leaf",
        "far",
    ),
    (
        "american",
        "tacos",
        "Mexican corn or flour tortillas filled with seasoned meat, fresh salsa, "
        "cilantro, and lime.",
        "corn tortillas | ground beef or chicken | onion | cilantro | lime juice | "
        "salsa | shredded cheese | sour cream | jalapeno",
        "far",
    ),
    (
        "american",
        "dumplings",
        "Steamed or pan-fried dough pockets stuffed with seasoned pork and cabbage, "
        "a staple of Chinese and East Asian cuisine.",
        "all-purpose flour | ground pork | napa cabbage | ginger | garlic | "
        "soy sauce | sesame oil | green onions | water",
        "far",
    ),
    (
        "american",
        "naan",
        "A soft, leavened South Asian flatbread traditionally baked in a tandoor oven, "
        "commonly served with curries.",
        "all-purpose flour | active dry yeast | plain yogurt | warm milk | "
        "butter | garlic | salt | sugar",
        "far",
    ),

    # ============================================================
    # Chinese — 10 dataset-backed pairs
    # ============================================================
    (
        "chinese",
        "Easy Fried Rice",
        "This fried rice recipe only takes 15 minutes to cook and tastes just like "
        "you get at your favorite Chinese restaurant. Leftover rice, plus a couple of eggs, "
        "baby carrots, peas, and soy sauce is all you need.",
        "2/3 cup chopped baby carrots | 1/2 cup frozen green peas | "
        "2 tablespoons vegetable oil | 1 clove garlic, minced | 2 large eggs | "
        "3 cups leftover cooked and chilled white rice | "
        "1 tablespoon soy sauce | 2 teaspoons sesame oil",
        "close",
    ),
    (
        "chinese",
        "Vegetable Fried Rice",
        "This vegetable fried rice combines the nutty flavor of brown rice with "
        "the fresh taste of bell peppers, baby peas, and other vegetables.",
        "3 cups water | 1 1/2 cups quick-cooking brown rice | 2 tablespoons peanut oil | "
        "1 small yellow onion, chopped | 1 small green bell pepper, chopped | "
        "1 teaspoon minced garlic | 1/4 teaspoon red pepper flakes | "
        "3 green onions, thinly sliced | 3 tablespoons soy sauce | "
        "1 cup frozen petite peas | 2 teaspoons sesame oil | 1/4 cup roasted peanuts",
        "close",
    ),
    (
        "chinese",
        "Homemade Wonton Soup",
        "This is a recipe I have perfected on my own through the years. "
        "This recipe is one that my family members beg me to make all the time!",
        "1 bunch green onions | 6 fresh mushrooms, sliced | 1 pound ground pork | "
        "1 tablespoon sesame oil | 1 tablespoon soy sauce | 1 egg | "
        "1/4 cup dry bread crumbs | 1 (16 ounce) package wonton wrappers | "
        "8 cups chicken broth | 16 uncooked medium shrimp | "
        "1 medium head bok choy | 16 snow peas",
        "close",
    ),
    (
        "chinese",
        "Pork Lo Mein",
        "This pork lo mein recipe was inspired by another recipe, but I added more "
        "vegetables, ginger, and sesame oil. Add or remove veggies as you see fit.",
        "1 (8 ounce) package linguine | 1/3 cup low-sodium soy sauce | "
        "2 tablespoons rice vinegar | 2 teaspoons cornstarch | 1 teaspoon white sugar | "
        "1/2 teaspoon sesame oil | 2 tablespoons canola oil | 2 cups snap peas | "
        "1 small sweet onion | 1 (12 ounce) pork tenderloin, cut into thin strips | "
        "1 (8 ounce) package sliced white mushrooms | 1 medium red bell pepper | "
        "3 cloves garlic | 1/2 teaspoon chopped fresh ginger | 3 green onions, sliced",
        "close",
    ),
    (
        "chinese",
        "Kung Pao Chicken",
        "This tasty kung pao chicken is similar to what is served in Chinese restaurants. "
        "It's easy to make, and you can be as creative with the measurements as you want. "
        "The sauce reduces until nice and thick.",
        "2 tablespoons cornstarch dissolved in 2 tablespoons water | "
        "2 tablespoons white wine | 2 tablespoons soy sauce | 2 tablespoons sesame oil | "
        "1 pound skinless boneless chicken breast, cut into chunks | "
        "1 ounce hot chili paste | 2 teaspoons brown sugar | "
        "1 teaspoon distilled white vinegar | 1 (8 ounce) can water chestnuts | "
        "4 ounces chopped peanuts | 4 green onions, chopped | 1 tablespoon chopped garlic",
        "close",
    ),
    (
        "chinese",
        "Smash Burgers",
        "This smash burger recipe makes super juicy burgers with crispy edges. "
        "I prefer to cook these outdoors — they grill up very fast because of the high heat.",
        "4 hamburger buns | 2 tablespoons butter | 1 pound ground chuck beef (80% lean) | "
        "salt to taste | 4 slices American cheese | burger toppings of choice",
        "far",
    ),
    (
        "chinese",
        "Chicken Biryani",
        "Chicken biryani is a delicious Pakistani/Indian rice dish typically reserved "
        "for special occasions. It has a lengthy preparation, but the work is worth it.",
        "4 tablespoons vegetable oil | 2 large onions, finely chopped | "
        "2 cloves garlic | 1 tablespoon minced fresh ginger | 1 teaspoon ground cumin | "
        "1/2 teaspoon chili powder | 1/2 teaspoon ground turmeric | "
        "2 tablespoons plain yogurt | 1/2 teaspoon ground cardamom | "
        "3 pounds boneless chicken pieces | 1 pound basmati rice | "
        "1 pinch powdered saffron | 4 cups chicken stock",
        "far",
    ),
    (
        "chinese",
        "Italian Pizza Dough",
        "My home-made pizza dough; I make it 3 to 4 times a week.",
        "1 1/2 teaspoons active dry yeast | 1/2 teaspoon white sugar | "
        "1/2 cup lukewarm water | 2 cups sifted all-purpose flour | "
        "1/2 teaspoon salt | 2 tablespoons olive oil | 1 egg, beaten",
        "far",
    ),
    (
        "chinese",
        "Shredded Pork Fajita Tacos",
        "A very quick and easy recipe. My family of 7 likes it so much "
        "that we have it a few times a month.",
        "1/4 cup butter | 1 onion, chopped | 1 green bell pepper, chopped | "
        "1 (16 ounce) package shredded pork | 1/4 cup water | "
        "1 (1 ounce) package fajita seasoning | 6 (6 inch) corn tortillas | "
        "1 cup shredded lettuce | 1 cup tomatoes, chopped | "
        "1 cup shredded Cheddar cheese | salsa | sour cream",
        "far",
    ),
    (
        "chinese",
        "Garlic Naan",
        "This homemade garlic naan recipe is cooked in a hot cast iron skillet. "
        "When cooked at the proper temperature, naan develops blistered bubbles "
        "with a lovely golden-black char.",
        "1/2 cup warm water | 1 teaspoon white sugar | "
        "1 package active dry yeast | 1/4 cup butter | 2 cloves garlic, minced | "
        "2 cups bread flour | 1/4 cup plain yogurt | 1 teaspoon kosher salt | "
        "1/4 cup chopped cilantro",
        "far",
    ),

    # Chinese — 10 general pairs
    (
        "chinese",
        "fried rice",
        "A Chinese stir-fried rice dish cooked in a wok with eggs, vegetables, "
        "and seasoned with soy sauce and sesame oil.",
        "cooked white rice | eggs | soy sauce | vegetable oil | garlic | "
        "green onions | frozen peas | carrots | sesame oil",
        "close",
    ),
    (
        "chinese",
        "wonton soup",
        "A Chinese broth-based soup with pork-filled wonton dumplings, "
        "bok choy, and spring onions.",
        "wonton wrappers | ground pork | shrimp | ginger | soy sauce | sesame oil | "
        "chicken broth | bok choy | green onions | salt | pepper",
        "close",
    ),
    (
        "chinese",
        "lo mein",
        "Soft Chinese egg noodles stir-fried with vegetables and a savory soy-sesame sauce.",
        "lo mein noodles | soy sauce | oyster sauce | sesame oil | garlic | ginger | "
        "bean sprouts | carrots | green onions | vegetable oil",
        "close",
    ),
    (
        "chinese",
        "dumplings",
        "Pan-fried or steamed Chinese dough pockets filled with seasoned pork and napa cabbage.",
        "all-purpose flour | ground pork | napa cabbage | ginger | garlic | "
        "soy sauce | sesame oil | green onions | water",
        "close",
    ),
    (
        "chinese",
        "kung pao chicken",
        "A classic Sichuan stir-fry of diced chicken with peanuts, dried chili peppers, "
        "and a sweet-savory sauce.",
        "chicken breast | peanuts | dried chili peppers | soy sauce | rice vinegar | "
        "cornstarch | garlic | ginger | vegetable oil | green onions | Sichuan peppercorns",
        "close",
    ),
    (
        "chinese",
        "burger",
        "A classic American sandwich featuring a grilled beef patty served in a bun "
        "with lettuce, tomato, onion, and condiments.",
        "ground beef | burger bun | lettuce | tomato | onion | pickles | "
        "ketchup | mustard | American cheese",
        "far",
    ),
    (
        "chinese",
        "pizza",
        "An Italian flatbread topped with tomato sauce, fresh mozzarella, and various "
        "toppings, baked in a hot oven.",
        "pizza dough | tomato sauce | mozzarella cheese | olive oil | garlic | "
        "fresh basil | oregano | toppings of choice",
        "far",
    ),
    (
        "chinese",
        "biryani",
        "A fragrant South Asian layered rice dish cooked with aromatic spices, "
        "marinated meat, and caramelized onions.",
        "basmati rice | chicken | onion | plain yogurt | garam masala | turmeric | "
        "cumin | coriander | saffron | ghee | fresh mint | bay leaf",
        "far",
    ),
    (
        "chinese",
        "tacos",
        "Mexican corn or flour tortillas filled with seasoned meat, fresh salsa, "
        "cilantro, and lime.",
        "corn tortillas | ground beef or chicken | onion | cilantro | lime juice | "
        "salsa | shredded cheese | sour cream | jalapeno",
        "far",
    ),
    (
        "chinese",
        "mac and cheese",
        "Creamy baked macaroni pasta smothered in a rich cheddar cheese sauce, "
        "the ultimate American comfort food.",
        "elbow macaroni | cheddar cheese | butter | whole milk | all-purpose flour | "
        "salt | black pepper | mustard powder | breadcrumbs",
        "far",
    ),

    # ============================================================
    # Indian — 10 dataset-backed pairs
    # ============================================================
    (
        "indian",
        "Chicken Biryani",
        "Chicken biryani is a delicious Pakistani/Indian rice dish that's typically reserved "
        "for special occasions such as weddings, parties, or holidays such as Ramadan. "
        "It has a lengthy preparation, but the work is definitely worth it.",
        "4 tablespoons vegetable oil | 4 small potatoes, peeled and halved | "
        "2 large onions, finely chopped | 2 cloves garlic, minced | "
        "1 tablespoon minced fresh ginger | 2 medium tomatoes, peeled and chopped | "
        "1 teaspoon salt | 1 teaspoon ground cumin | 1/2 teaspoon chili powder | "
        "1/2 teaspoon ground turmeric | 2 tablespoons plain yogurt | "
        "2 tablespoons chopped fresh mint | 1/2 teaspoon ground cardamom | "
        "3 pounds boneless chicken pieces | 1 pound basmati rice | "
        "1 pinch powdered saffron | 4 cups chicken stock",
        "close",
    ),
    (
        "indian",
        "Lamb (Gosht) Biryani",
        "This festive lamb biryani dish is perfect for gatherings or celebrations. "
        "Biryani is a bit of a project and is time-consuming, but I have never been "
        "disappointed with the results.",
        "2 1/2 cups basmati rice | 1/4 cup cooking oil | 8 whole cloves | "
        "4 black cardamom pods | 4 cinnamon sticks | 4 large onions, sliced thin | "
        "1 tablespoon garlic paste | 1 tablespoon ginger paste | "
        "1/4 cup chopped fresh cilantro | 3 tablespoons chopped fresh mint | "
        "1 pound lamb chops | salt to taste | 3 tomatoes, chopped | "
        "4 green chile peppers | 2 teaspoons ground red pepper | "
        "2 tablespoons plain yogurt | 2 tablespoons lemon juice | "
        "1/2 teaspoon saffron | 2 tablespoons warm milk",
        "close",
    ),
    (
        "indian",
        "Tandoori Chicken",
        "Try this authentic tandoori chicken that's marinated in yogurt and spices, "
        "then cooked on the grill instead of a clay oven so you can make it at home.",
        "2 pounds chicken, cut into pieces | 1 medium lemon, juiced | 1 teaspoon salt | "
        "1 1/4 cups plain yogurt | 1/2 medium onion, finely chopped | "
        "1 clove garlic, minced | 2 teaspoons garam masala | "
        "1 teaspoon grated fresh ginger | 1 teaspoon cayenne pepper | "
        "2 teaspoons finely chopped cilantro | 1 medium lemon, cut into wedges",
        "close",
    ),
    (
        "indian",
        "Paneer Tikka Masala",
        "This paneer tikka masala made with cubes of paneer cheese in a spicy creamy curry sauce "
        "is easy to make. This vegetarian dish goes well with naan or basmati rice.",
        "1/4 cup butter | 1 pound paneer, cut into 1/2-inch cubes | "
        "2 medium onions, finely chopped | 1 medium green bell pepper | "
        "2 medium jalapeno peppers | 1 tablespoon ground cashews | "
        "1 teaspoon garlic paste | 1 teaspoon ginger paste | 1 teaspoon cayenne pepper | "
        "1 teaspoon ground cumin | 1 teaspoon garam masala | "
        "1 (16 ounce) can tomato sauce | 1 pint half-and-half | 1 teaspoon salt",
        "close",
    ),
    (
        "indian",
        "Garlic Naan",
        "This homemade garlic naan recipe is cooked in a hot cast iron skillet. "
        "When cooked at the proper temperature, naan develops blistered bubbles "
        "with a lovely golden-black char. Reward yourself by sopping the bread in curry.",
        "1/2 cup warm water | 1 teaspoon white sugar | 1 package active dry yeast | "
        "1/4 cup butter | 2 cloves garlic, minced | 2 cups bread flour | "
        "1/4 cup plain yogurt | 1 teaspoon kosher salt | 1/4 cup chopped cilantro",
        "close",
    ),
    (
        "indian",
        "Smash Burgers",
        "This smash burger recipe makes super juicy burgers with crispy edges. "
        "I prefer to cook these outdoors — they grill up very fast because of the high heat.",
        "4 hamburger buns | 2 tablespoons butter | 1 pound ground chuck beef (80% lean) | "
        "salt to taste | 4 slices American cheese | burger toppings of choice",
        "far",
    ),
    (
        "indian",
        "Easy Fried Rice",
        "This fried rice recipe only takes 15 minutes to cook and tastes just like "
        "you get at your favorite Chinese restaurant.",
        "2/3 cup chopped baby carrots | 1/2 cup frozen green peas | "
        "2 tablespoons vegetable oil | 1 clove garlic | 2 large eggs | "
        "3 cups leftover cooked white rice | 1 tablespoon soy sauce | 2 teaspoons sesame oil",
        "far",
    ),
    (
        "indian",
        "Cheese Lasagna",
        "This cheese lasagna recipe is an easy vegetarian lasagna with "
        "ricotta, mozzarella, and Parmesan cheeses.",
        "1 (16 ounce) package lasagna noodles | 4 cups ricotta cheese | 4 eggs | "
        "1/4 cup grated Parmesan cheese | salt and pepper to taste | 1 teaspoon olive oil | "
        "3 cloves garlic, minced | 1 (32 ounce) jar spaghetti sauce | "
        "1 teaspoon Italian seasoning | 2 cups shredded mozzarella cheese",
        "far",
    ),
    (
        "indian",
        "Barbacoa Tacos",
        "These barbacoa tacos are packed with smoky shredded beef that's perfectly tender. "
        "Spices like cumin complement the chiles, while oregano and bay leaves add earthiness.",
        "2 ripe plum tomatoes | 1 small white onion, quartered | 2 cloves garlic | "
        "4 chipotle peppers in adobo sauce | 1 teaspoon ground cumin | "
        "1 (3 pound) beef chuck roast | 2 tablespoons olive oil | "
        "2 teaspoons dried oregano | 3 fresh bay leaves | 1 tablespoon lime juice | "
        "corn tortillas | 2 ripe avocados | fresh cilantro",
        "far",
    ),
    (
        "indian",
        "Chicago-Style Hot Dog",
        "This hot dog is a Windy City classic and a big favorite with sports fans! "
        "The frank must be all-beef, the bun must be poppy seed.",
        "1 all-beef hot dog | 1 poppy seed hot dog bun | 1 tablespoon yellow mustard | "
        "1 tablespoon sweet green pickle relish | 1 tablespoon chopped onion | "
        "2 tomato wedges | 1 dill pickle spear | 2 sport peppers | 1 dash celery salt",
        "far",
    ),

    # Indian — 10 general pairs
    (
        "indian",
        "biryani",
        "A fragrant South Asian layered rice dish cooked with aromatic whole spices, "
        "marinated meat, and caramelized onions, finished with saffron.",
        "basmati rice | chicken | onion | plain yogurt | garam masala | turmeric | "
        "cumin | coriander | saffron | ghee | fresh mint | bay leaf | cardamom | cloves",
        "close",
    ),
    (
        "indian",
        "tandoori chicken",
        "Yogurt-marinated chicken cooked in a clay tandoor oven with aromatic Indian spices, "
        "giving it a smoky char and vibrant color.",
        "chicken pieces | plain yogurt | lemon juice | tandoori masala | cumin | "
        "coriander | paprika | ginger | garlic | salt | red food coloring",
        "close",
    ),
    (
        "indian",
        "samosa",
        "A crispy deep-fried pastry filled with spiced potatoes, peas, and herbs — "
        "a popular Indian street food snack.",
        "all-purpose flour | potato | green peas | cumin seeds | coriander | "
        "fresh ginger | green chili | garam masala | amchur | oil for frying | salt",
        "close",
    ),
    (
        "indian",
        "paneer",
        "Fresh Indian cottage cheese made from whole milk, used in curries, grilled dishes, "
        "or stuffed into flatbreads.",
        "whole milk | lemon juice or white vinegar | salt",
        "close",
    ),
    (
        "indian",
        "naan",
        "A soft leavened Indian flatbread traditionally baked in a tandoor oven, "
        "commonly served alongside curries and dals.",
        "all-purpose flour | active dry yeast | plain yogurt | warm milk | "
        "butter | garlic | salt | sugar",
        "close",
    ),
    (
        "indian",
        "sushi",
        "A Japanese dish of vinegared rice paired with fresh fish, seafood, or vegetables, "
        "served with soy sauce, wasabi, and pickled ginger.",
        "sushi rice | nori | salmon | tuna | cucumber | avocado | soy sauce | "
        "wasabi | pickled ginger | rice vinegar | sugar | salt",
        "far",
    ),
    (
        "indian",
        "burger",
        "A classic American sandwich featuring a grilled beef patty served in a bun "
        "with lettuce, tomato, onion, and condiments.",
        "ground beef | burger bun | lettuce | tomato | onion | pickles | "
        "ketchup | mustard | American cheese",
        "far",
    ),
    (
        "indian",
        "hot dog",
        "An American staple: a pork or beef frankfurter served in a soft bun, "
        "topped with mustard, ketchup, and relish.",
        "hot dog sausage | hot dog bun | yellow mustard | ketchup | sweet relish | onion",
        "far",
    ),
    (
        "indian",
        "lasagna",
        "Classic Italian baked pasta with alternating layers of meat sauce, "
        "bechamel, and melted cheese.",
        "lasagna noodles | ground beef | tomato sauce | ricotta cheese | "
        "mozzarella cheese | Parmesan cheese | onion | garlic | olive oil | fresh basil",
        "far",
    ),
    (
        "indian",
        "tacos",
        "Mexican corn or flour tortillas filled with seasoned meat, fresh salsa, "
        "cilantro, and lime.",
        "corn tortillas | ground beef or chicken | onion | cilantro | lime juice | "
        "salsa | shredded cheese | sour cream | jalapeno",
        "far",
    ),

    # ============================================================
    # Italian — 10 dataset-backed pairs
    # ============================================================
    (
        "italian",
        "White Cheese Chicken Lasagna",
        "Chicken lasagna with spinach and a creamy white cheese sauce. "
        "Great for any kind of potluck. My kids love it!",
        "9 lasagna noodles | 1/2 cup butter | 1 onion, chopped | 1 clove garlic, minced | "
        "1/2 cup all-purpose flour | 2 cups chicken broth | 1 1/2 cups milk | 1 teaspoon salt | "
        "4 cups shredded mozzarella cheese | 1 cup grated Parmesan cheese | "
        "1 teaspoon dried basil | 1 teaspoon dried oregano | 2 cups ricotta cheese | "
        "2 cups cubed cooked chicken | 2 packages frozen chopped spinach",
        "close",
    ),
    (
        "italian",
        "Cheese Lasagna",
        "This cheese lasagna recipe is an easy vegetarian lasagna with "
        "ricotta, mozzarella, and Parmesan cheeses.",
        "1 (16 ounce) package lasagna noodles | 4 cups ricotta cheese | 4 eggs | "
        "1/4 cup grated Parmesan cheese | salt and pepper to taste | 1 teaspoon olive oil | "
        "3 cloves garlic, minced | 1 (32 ounce) jar spaghetti sauce | "
        "1 teaspoon Italian seasoning | 2 cups shredded mozzarella cheese",
        "close",
    ),
    (
        "italian",
        "Linguine with Clams",
        "This is one of my favorite meals. This goes great with a salad and some garlic bread.",
        "1 (16 ounce) package linguine pasta | 8 tablespoons unsalted butter | "
        "1 medium white onion, chopped | 8 ounces fresh mushrooms, sliced | "
        "4 cloves garlic, pressed | 1 cup dry white wine | "
        "4 (6.5 ounce) cans chopped clams, drained with juices reserved | "
        "2 tablespoons sour cream | freshly ground black pepper | "
        "1/4 cup chopped flat leaf parsley",
        "close",
    ),
    (
        "italian",
        "Mushroom and Pea Risotto",
        "My very first risotto recipe and I must say, it's pretty legit. "
        "Rich and creamy with loads of mushrooms, peas, and cheese.",
        "4 cups vegetable broth | 3 tablespoons extra-virgin olive oil | "
        "2 tablespoons unsalted butter | 1 medium yellow onion, minced | "
        "4 cloves garlic | 1/2 teaspoon red pepper flakes | "
        "2 cups cremini mushrooms, sliced | 3/4 cup shiitake mushrooms, sliced | "
        "1 teaspoon chopped fresh thyme | 2 cups Arborio rice | 1 large bay leaf | "
        "1/2 cup dry white wine | 1 cup frozen peas | "
        "1 cup shredded Pecorino Romano cheese | 1/3 cup chopped fresh parsley",
        "close",
    ),
    (
        "italian",
        "Italian Tiramisu",
        "Tiramisu is a classic Italian dessert. Ladyfinger cookies are dipped in coffee, "
        "then layered with mascarpone and dusted with cocoa powder.",
        "6 egg yolks | 1 cup white sugar | 1 pound mascarpone cheese | "
        "6 egg whites, stiffly beaten | 1/4 cup heavy cream | 3 tablespoons kirschwasser | "
        "1 1/4 cups strong brewed coffee, cold | 25 ladyfingers | "
        "1 tablespoon unsweetened cocoa powder",
        "close",
    ),
    (
        "italian",
        "Chicken Biryani",
        "Chicken biryani is a delicious Pakistani/Indian rice dish typically reserved "
        "for special occasions. It has a lengthy preparation but the work is worth it.",
        "4 tablespoons vegetable oil | 2 large onions, finely chopped | "
        "2 cloves garlic | 1 tablespoon minced fresh ginger | 1 teaspoon ground cumin | "
        "1/2 teaspoon chili powder | 1/2 teaspoon ground turmeric | "
        "2 tablespoons plain yogurt | 1/2 teaspoon ground cardamom | "
        "3 pounds boneless chicken pieces | 1 pound basmati rice | "
        "1 pinch powdered saffron | 4 cups chicken stock",
        "far",
    ),
    (
        "italian",
        "Homemade Wonton Soup",
        "This is a recipe I have perfected on my own through the years. "
        "This recipe is one that my family members beg me to make all the time!",
        "1 bunch green onions | 6 fresh mushrooms, sliced | 1 pound ground pork | "
        "1 tablespoon sesame oil | 1 tablespoon soy sauce | 1 egg | "
        "1 (16 ounce) package wonton wrappers | 8 cups chicken broth | "
        "1 medium head bok choy | 16 snow peas",
        "far",
    ),
    (
        "italian",
        "Smash Burgers",
        "This smash burger recipe makes super juicy burgers with crispy edges. "
        "I prefer to cook these outdoors — they grill up very fast because of the high heat.",
        "4 hamburger buns | 2 tablespoons butter | 1 pound ground chuck beef (80% lean) | "
        "salt to taste | 4 slices American cheese | burger toppings of choice",
        "far",
    ),
    (
        "italian",
        "Barbacoa Tacos",
        "These barbacoa tacos are packed with smoky shredded beef that's perfectly tender. "
        "Spices like cumin complement the chiles, while oregano and bay leaves add earthiness.",
        "2 ripe plum tomatoes | 1 small white onion | 4 chipotle peppers in adobo sauce | "
        "1 teaspoon ground cumin | 1 (3 pound) beef chuck roast | "
        "2 teaspoons dried oregano | 3 fresh bay leaves | 1 tablespoon lime juice | "
        "corn tortillas | 2 ripe avocados | fresh cilantro",
        "far",
    ),
    (
        "italian",
        "Tandoori Chicken",
        "Authentic tandoori chicken marinated in yogurt and spices, "
        "then cooked on the grill instead of a clay oven.",
        "2 pounds chicken, cut into pieces | 1 medium lemon, juiced | 1 teaspoon salt | "
        "1 1/4 cups plain yogurt | 1 clove garlic, minced | 2 teaspoons garam masala | "
        "1 teaspoon grated fresh ginger | 1 teaspoon cayenne pepper | "
        "2 teaspoons finely chopped cilantro",
        "far",
    ),

    # Italian — 10 general pairs
    (
        "italian",
        "pasta",
        "Italian pasta cooked al dente and tossed with a rich tomato or cream-based sauce, "
        "finished with Parmesan and fresh herbs.",
        "pasta | tomato sauce | garlic | olive oil | onion | Parmesan cheese | "
        "fresh basil | salt | black pepper",
        "close",
    ),
    (
        "italian",
        "lasagna",
        "Classic Italian baked pasta with alternating layers of meat sauce, "
        "bechamel, and melted cheese, baked until bubbly.",
        "lasagna noodles | ground beef | tomato sauce | ricotta cheese | "
        "mozzarella cheese | Parmesan cheese | onion | garlic | olive oil | fresh basil",
        "close",
    ),
    (
        "italian",
        "risotto",
        "A creamy Italian rice dish slow-cooked with ladles of hot broth, "
        "white wine, and finished with butter and Parmesan.",
        "Arborio rice | chicken or vegetable broth | dry white wine | onion | garlic | "
        "unsalted butter | Parmesan cheese | olive oil | salt | black pepper",
        "close",
    ),
    (
        "italian",
        "pizza",
        "Italian flatbread topped with tomato sauce, fresh mozzarella, and ingredients "
        "like basil, prosciutto, or vegetables, baked in a hot oven.",
        "pizza dough | tomato sauce | fresh mozzarella | olive oil | fresh basil | "
        "garlic | dried oregano",
        "close",
    ),
    (
        "italian",
        "tiramisu",
        "A classic Italian no-bake dessert of espresso-soaked ladyfingers layered with "
        "mascarpone cream and dusted with cocoa powder.",
        "ladyfingers | mascarpone cheese | espresso | egg yolks | white sugar | "
        "heavy cream | unsweetened cocoa powder | rum or marsala wine",
        "close",
    ),
    (
        "italian",
        "biryani",
        "A fragrant South Asian layered rice dish cooked with aromatic whole spices, "
        "marinated meat, and caramelized onions.",
        "basmati rice | chicken | onion | plain yogurt | garam masala | turmeric | "
        "cumin | coriander | saffron | ghee | fresh mint | bay leaf",
        "far",
    ),
    (
        "italian",
        "sushi",
        "A Japanese dish of vinegared rice paired with fresh fish, seafood, or vegetables, "
        "served with soy sauce, wasabi, and pickled ginger.",
        "sushi rice | nori | salmon | tuna | cucumber | avocado | soy sauce | "
        "wasabi | pickled ginger | rice vinegar",
        "far",
    ),
    (
        "italian",
        "dumplings",
        "Steamed or pan-fried dough pockets stuffed with seasoned pork and cabbage, "
        "a staple of Chinese and East Asian cuisine.",
        "all-purpose flour | ground pork | napa cabbage | ginger | garlic | "
        "soy sauce | sesame oil | green onions | water",
        "far",
    ),
    (
        "italian",
        "tacos",
        "Mexican corn or flour tortillas filled with seasoned meat, fresh salsa, "
        "cilantro, and lime.",
        "corn tortillas | ground beef or chicken | onion | cilantro | lime juice | "
        "salsa | shredded cheese | sour cream | jalapeno",
        "far",
    ),
    (
        "italian",
        "naan",
        "A soft leavened South Asian flatbread traditionally baked in a tandoor oven, "
        "commonly served with curries.",
        "all-purpose flour | active dry yeast | plain yogurt | warm milk | "
        "butter | garlic | salt | sugar",
        "far",
    ),

    # ============================================================
    # Mexican — 10 dataset-backed pairs
    # ============================================================
    (
        "mexican",
        "Shredded Pork Fajita Tacos",
        "A very quick and easy recipe using leftovers. My family of 7 likes it "
        "so much that we have it a few times a month.",
        "1/4 cup butter | 1 onion, chopped | 1 green bell pepper, chopped | "
        "1 (16 ounce) package shredded pork | 1/4 cup water | "
        "1 (1 ounce) package fajita seasoning | 6 (6 inch) corn tortillas | "
        "1 cup shredded lettuce | 1 cup tomatoes, chopped | "
        "1 cup shredded Cheddar cheese | salsa | sour cream",
        "close",
    ),
    (
        "mexican",
        "Favorite Fry Bread Tacos",
        "These fry bread tacos are a favorite at aboriginal pow wows all summer long. "
        "For easier eating, cut cooked fry bread into crouton-sized pieces.",
        "2 cups all-purpose flour | 1 tablespoon baking powder | 1/2 teaspoon white sugar | "
        "1/2 teaspoon salt | 1 1/2 cups lukewarm water | 2 cups oil for frying | "
        "1 pound ground beef | 1 (15 ounce) can kidney beans, drained | "
        "1 package chili seasoning mix | 2 cups shredded Cheddar cheese | "
        "2 cups chopped iceberg lettuce | 2 tomatoes, chopped | 1 cup sour cream",
        "close",
    ),
    (
        "mexican",
        "Barbacoa Tacos",
        "These barbacoa tacos are packed with smoky shredded beef that's perfectly tender. "
        "Spices like cumin complement the chiles, while oregano and bay leaves add earthiness.",
        "2 ripe plum tomatoes | 1 small white onion, quartered | 2 cloves garlic | "
        "4 chipotle peppers in adobo sauce | 3 teaspoons kosher salt | "
        "1 1/2 teaspoons chili powder | 1 teaspoon ground cumin | "
        "1 (3 pound) beef chuck roast | 2 tablespoons olive oil | "
        "2 teaspoons dried oregano | 3 fresh bay leaves | 1 tablespoon lime juice | "
        "corn tortillas | 2 ripe avocados | fresh cilantro",
        "close",
    ),
    (
        "mexican",
        "Cottage Cheese Chicken Enchiladas",
        "Ever tried chicken enchiladas made with cottage cheese? Now's your chance! "
        "This takes some prep time, but it is well worth it.",
        "1 tablespoon vegetable oil | 2 skinless boneless chicken breast halves, shredded | "
        "1/2 cup chopped onion | 1 (7 ounce) can chopped green chile peppers | "
        "1 package taco seasoning mix | 1/2 cup sour cream | 2 cups cottage cheese | "
        "1 teaspoon salt | 12 (6 inch) corn tortillas | "
        "2 cups shredded Monterey Jack cheese | 1 (10 ounce) can red enchilada sauce",
        "close",
    ),
    (
        "mexican",
        "LuvAnn's Guacamole",
        "This homemade guacamole recipe is simple and always the first to get eaten "
        "at our barbecues. Perfect with tortilla chips or as a topping for tacos.",
        "2 medium avocados, peeled, pitted and diced | 1 1/2 teaspoons salt | "
        "1 large tomato, diced | 1 medium onion, diced | "
        "2 medium jalapeno peppers, chopped | 2 tablespoons fresh lime juice | "
        "1/2 tablespoon chopped fresh cilantro",
        "close",
    ),
    (
        "mexican",
        "Chicken Biryani",
        "Chicken biryani is a delicious Pakistani/Indian rice dish typically reserved "
        "for special occasions. It has a lengthy preparation but the work is worth it.",
        "4 tablespoons vegetable oil | 2 large onions, finely chopped | "
        "2 cloves garlic | 1 tablespoon minced fresh ginger | 1 teaspoon ground cumin | "
        "1/2 teaspoon chili powder | 1/2 teaspoon ground turmeric | "
        "2 tablespoons plain yogurt | 3 pounds boneless chicken pieces | "
        "1 pound basmati rice | 1 pinch powdered saffron | 4 cups chicken stock",
        "far",
    ),
    (
        "mexican",
        "Easy Fried Rice",
        "This fried rice recipe only takes 15 minutes to cook and tastes just like "
        "you get at your favorite Chinese restaurant.",
        "2/3 cup chopped baby carrots | 1/2 cup frozen green peas | "
        "2 tablespoons vegetable oil | 1 clove garlic | 2 large eggs | "
        "3 cups leftover cooked white rice | 1 tablespoon soy sauce | 2 teaspoons sesame oil",
        "far",
    ),
    (
        "mexican",
        "Cheese Lasagna",
        "This cheese lasagna recipe is an easy vegetarian lasagna with "
        "ricotta, mozzarella, and Parmesan cheeses.",
        "1 (16 ounce) package lasagna noodles | 4 cups ricotta cheese | 4 eggs | "
        "1/4 cup grated Parmesan cheese | salt and pepper to taste | 1 teaspoon olive oil | "
        "3 cloves garlic | 1 (32 ounce) jar spaghetti sauce | "
        "1 teaspoon Italian seasoning | 2 cups shredded mozzarella cheese",
        "far",
    ),
    (
        "mexican",
        "Smash Burgers",
        "This smash burger recipe makes super juicy burgers with crispy edges. "
        "I prefer to cook these outdoors — they grill up very fast because of the high heat.",
        "4 hamburger buns | 2 tablespoons butter | 1 pound ground chuck beef (80% lean) | "
        "salt to taste | 4 slices American cheese | burger toppings of choice",
        "far",
    ),
    (
        "mexican",
        "Tandoori Chicken",
        "Authentic tandoori chicken marinated in yogurt and spices, "
        "then cooked on the grill instead of a clay oven.",
        "2 pounds chicken, cut into pieces | 1 medium lemon, juiced | 1 teaspoon salt | "
        "1 1/4 cups plain yogurt | 1 clove garlic | 2 teaspoons garam masala | "
        "1 teaspoon grated fresh ginger | 1 teaspoon cayenne pepper | "
        "2 teaspoons finely chopped cilantro",
        "far",
    ),

    # Mexican — 10 general pairs
    (
        "mexican",
        "tacos",
        "Corn or flour tortillas filled with seasoned meat, fresh salsa, onion, "
        "cilantro, and a squeeze of lime.",
        "corn tortillas | ground beef or chicken | onion | fresh cilantro | "
        "lime juice | salsa verde | shredded cheese | sour cream | jalapeno",
        "close",
    ),
    (
        "mexican",
        "burrito",
        "A large flour tortilla stuffed with seasoned rice, black beans, meat, "
        "cheese, sour cream, and salsa.",
        "large flour tortilla | cooked rice | black beans | seasoned ground beef or chicken | "
        "shredded cheese | sour cream | salsa | guacamole | jalapeno",
        "close",
    ),
    (
        "mexican",
        "enchiladas",
        "Corn tortillas rolled with shredded chicken or beef and cheese, "
        "smothered in red or green enchilada sauce and baked.",
        "corn tortillas | shredded chicken or beef | red enchilada sauce | "
        "shredded cheese | onion | sour cream | fresh cilantro | jalapeno",
        "close",
    ),
    (
        "mexican",
        "guacamole",
        "A creamy Mexican dip made from mashed ripe avocados, lime juice, "
        "tomato, onion, cilantro, and jalapeno.",
        "ripe avocado | lime juice | tomato | white onion | fresh cilantro | "
        "jalapeno | garlic | salt",
        "close",
    ),
    (
        "mexican",
        "quesadilla",
        "A flour tortilla filled with melted cheese and optional fillings like "
        "chicken or peppers, grilled until golden and crispy.",
        "large flour tortilla | shredded cheese | cooked chicken or beef | "
        "bell pepper | onion | sour cream | salsa | butter or oil",
        "close",
    ),
    (
        "mexican",
        "biryani",
        "A fragrant South Asian layered rice dish cooked with aromatic whole spices, "
        "marinated meat, and caramelized onions.",
        "basmati rice | chicken | onion | plain yogurt | garam masala | turmeric | "
        "cumin | coriander | saffron | ghee | fresh mint | bay leaf",
        "far",
    ),
    (
        "mexican",
        "sushi",
        "A Japanese dish of vinegared rice paired with fresh fish, seafood, or vegetables, "
        "served with soy sauce, wasabi, and pickled ginger.",
        "sushi rice | nori | salmon | tuna | cucumber | avocado | soy sauce | "
        "wasabi | pickled ginger | rice vinegar",
        "far",
    ),
    (
        "mexican",
        "lasagna",
        "Classic Italian baked pasta with alternating layers of meat sauce, "
        "bechamel, and melted cheese.",
        "lasagna noodles | ground beef | tomato sauce | ricotta cheese | "
        "mozzarella cheese | Parmesan cheese | onion | garlic | olive oil | fresh basil",
        "far",
    ),
    (
        "mexican",
        "wonton soup",
        "A Chinese broth-based soup with pork-filled wonton dumplings, "
        "bok choy, and spring onions.",
        "wonton wrappers | ground pork | shrimp | ginger | soy sauce | sesame oil | "
        "chicken broth | bok choy | green onions | salt | pepper",
        "far",
    ),
    (
        "mexican",
        "ramen",
        "A Japanese noodle soup with a rich pork or chicken broth, "
        "wheat noodles, and toppings like chashu pork and soft-boiled egg.",
        "ramen noodles | pork or chicken broth | soy sauce | miso paste | "
        "chashu pork belly | soft-boiled egg | nori | bamboo shoots | "
        "green onions | sesame oil | bean sprouts",
        "far",
    ),

    # ============================================================
    # Other — 10 dataset-backed pairs
    # ============================================================
    (
        "other",
        "Chicken Souvlaki with Tzatziki Sauce",
        "Chicken souvlaki skewers are marinated Greek kabobs. "
        "Fantastic flavor for chicken. The marinade can also be used for pork.",
        "1/4 cup olive oil | 2 cloves garlic, minced | 2 tablespoons lemon juice | "
        "1 teaspoon dried oregano | 1/2 teaspoon salt | "
        "1 1/2 pounds skinless boneless chicken breast, cut into bite-sized pieces | "
        "6 wooden skewers | 1 (6 ounce) container plain Greek-style yogurt | "
        "1/2 cucumber, peeled, seeded, and grated | 1 tablespoon olive oil | "
        "2 teaspoons white vinegar | 1 clove garlic, minced | 1 pinch salt",
        "close",
    ),
    (
        "other",
        "Traditional Gyros",
        "Make a Greek gyro sandwich at home with this recipe that I absolutely love!",
        "1 small onion, cut into chunks | 1 pound ground lamb | 1 pound ground beef | "
        "1 tablespoon minced garlic | 1 teaspoon dried oregano | 1 teaspoon ground cumin | "
        "1 teaspoon dried marjoram | 1 teaspoon dried thyme | 1 teaspoon dried rosemary | "
        "1 teaspoon freshly ground black pepper | 1/4 teaspoon sea salt | "
        "12 tablespoons hummus | 12 pita bread rounds | 1 small head lettuce, shredded | "
        "1 large tomato, sliced | 1 large red onion, sliced | "
        "6 ounces crumbled feta cheese | 24 tablespoons tzatziki sauce",
        "close",
    ),
    (
        "other",
        "Spicy Three Pepper Hummus",
        "My family loves a popular brand of 3 pepper hummus. This recipe makes 3 times "
        "the amount in store-bought containers for about the same price.",
        "2 (16 ounce) cans garbanzo beans, drained | 2 tablespoons olive oil | "
        "1/8 cup lemon juice | 2 tablespoons tahini | 8 cloves garlic, minced | "
        "2 slices jarred jalapeno pepper, chopped | 1 teaspoon liquid from jalapeno jar | "
        "1/2 teaspoon ground black pepper | 1 1/2 teaspoons cayenne pepper | "
        "1/2 teaspoon ground cumin | 3/4 teaspoon dried oregano",
        "close",
    ),
    (
        "other",
        "Pad Thai with Tofu",
        "This is a favorite Thai dish that is light and combines sour, salt, sweet, and spicy flavors.",
        "1 (12 ounce) package tofu, drained and cubed | 1 tablespoon cornstarch | "
        "3 tablespoons vegetable oil | 8 ounces dry rice stick noodles | 1/4 cup water | "
        "1/4 cup sriracha hot sauce | 1/4 cup soy sauce | 2 tablespoons white sugar | "
        "1 tablespoon tamarind concentrate | 1 teaspoon red pepper flakes | "
        "1/2 onion, sliced | 1 egg | 2 tablespoons chopped spring onions | "
        "1 tablespoon crushed peanuts | 1 lime, cut into wedges",
        "close",
    ),
    (
        "other",
        "Authentic Seafood Paella",
        "Paella is a classic Spanish dish of rice cooked with shellfish and seasoned "
        "with saffron. It is perfect for easy entertaining.",
        "2 tablespoons olive oil | 1 onion, finely diced | 1/2 tomato, finely diced | "
        "1/2 tablespoon smoked paprika | 6 fresh romano beans | "
        "1/2 cup canned butter beans, drained | 1/2 cup white rice | "
        "6 large shrimp | 6 mussels | 6 clams | 1 cup white wine | "
        "2 cups seafood stock | 1 pinch saffron threads | "
        "1 teaspoon finely chopped fresh rosemary | 1 cup fresh peas | "
        "5 baby squid | 1 lemon | 1 tablespoon chopped fresh flat-leaf parsley",
        "close",
    ),
    (
        "other",
        "Smash Burgers",
        "This smash burger recipe makes super juicy burgers with crispy edges. "
        "I prefer to cook these outdoors — they grill up very fast because of the high heat.",
        "4 hamburger buns | 2 tablespoons butter | 1 pound ground chuck beef (80% lean) | "
        "salt to taste | 4 slices American cheese | burger toppings of choice",
        "far",
    ),
    (
        "other",
        "Chicken Biryani",
        "Chicken biryani is a delicious Pakistani/Indian rice dish typically reserved "
        "for special occasions. It has a lengthy preparation but the work is worth it.",
        "4 tablespoons vegetable oil | 2 large onions, finely chopped | "
        "2 cloves garlic | 1 tablespoon minced fresh ginger | 1 teaspoon ground cumin | "
        "1/2 teaspoon chili powder | 1/2 teaspoon ground turmeric | "
        "2 tablespoons plain yogurt | 3 pounds boneless chicken pieces | "
        "1 pound basmati rice | 1 pinch powdered saffron | 4 cups chicken stock",
        "far",
    ),
    (
        "other",
        "Easy Fried Rice",
        "This fried rice recipe only takes 15 minutes to cook and tastes just like "
        "you get at your favorite Chinese restaurant.",
        "2/3 cup chopped baby carrots | 1/2 cup frozen green peas | "
        "2 tablespoons vegetable oil | 1 clove garlic | 2 large eggs | "
        "3 cups leftover cooked white rice | 1 tablespoon soy sauce | 2 teaspoons sesame oil",
        "far",
    ),
    (
        "other",
        "White Cheese Chicken Lasagna",
        "Chicken lasagna with spinach and a creamy white cheese sauce. "
        "Great for any kind of potluck. My kids love it!",
        "9 lasagna noodles | 1/2 cup butter | 1 onion | 1 clove garlic | "
        "1/2 cup all-purpose flour | 2 cups chicken broth | 1 1/2 cups milk | "
        "4 cups shredded mozzarella cheese | 1 cup grated Parmesan cheese | "
        "1 teaspoon dried basil | 2 cups ricotta cheese | "
        "2 cups cubed cooked chicken | 2 packages frozen chopped spinach",
        "far",
    ),
    (
        "other",
        "Barbacoa Tacos",
        "These barbacoa tacos are packed with smoky shredded beef that's perfectly tender. "
        "Spices like cumin complement the chiles, while oregano and bay leaves add earthiness.",
        "2 ripe plum tomatoes | 1 small white onion | 4 chipotle peppers in adobo sauce | "
        "1 teaspoon ground cumin | 1 (3 pound) beef chuck roast | "
        "2 teaspoons dried oregano | 3 fresh bay leaves | 1 tablespoon lime juice | "
        "corn tortillas | 2 ripe avocados | fresh cilantro",
        "far",
    ),

    # Other — 10 general pairs
    (
        "other",
        "falafel",
        "Deep-fried Middle Eastern patties made from ground chickpeas, herbs, and spices, "
        "served in pita with tahini sauce.",
        "chickpeas | white onion | garlic | fresh parsley | fresh cilantro | "
        "ground cumin | ground coriander | all-purpose flour | baking soda | "
        "salt | oil for frying",
        "close",
    ),
    (
        "other",
        "hummus",
        "A creamy Middle Eastern dip made from blended chickpeas, tahini, lemon, "
        "and garlic, drizzled with olive oil.",
        "chickpeas | tahini | fresh lemon juice | garlic | olive oil | "
        "ground cumin | salt | cold water",
        "close",
    ),
    (
        "other",
        "pad thai",
        "A popular Thai stir-fried rice noodle dish with a tamarind-based sauce, "
        "eggs, bean sprouts, and crushed peanuts.",
        "rice stick noodles | shrimp or tofu | egg | bean sprouts | green onions | "
        "crushed peanuts | lime | tamarind paste | fish sauce | white sugar | vegetable oil",
        "close",
    ),
    (
        "other",
        "sushi",
        "A Japanese dish of vinegared rice paired with fresh fish, seafood, or vegetables, "
        "served with soy sauce, wasabi, and pickled ginger.",
        "sushi rice | nori | salmon | tuna | cucumber | avocado | soy sauce | "
        "wasabi | pickled ginger | rice vinegar | sugar | salt",
        "close",
    ),
    (
        "other",
        "paella",
        "A classic Spanish saffron-scented rice dish cooked in a wide shallow pan "
        "with seafood, chicken, or a mix of both.",
        "short-grain rice | shrimp | mussels | clams | chicken thighs | saffron threads | "
        "tomato | onion | garlic | smoked paprika | olive oil | chicken broth | lemon",
        "close",
    ),
    (
        "other",
        "burger",
        "A classic American sandwich featuring a grilled beef patty served in a bun "
        "with lettuce, tomato, onion, and condiments.",
        "ground beef | burger bun | lettuce | tomato | onion | pickles | "
        "ketchup | mustard | American cheese",
        "far",
    ),
    (
        "other",
        "biryani",
        "A fragrant South Asian layered rice dish cooked with aromatic whole spices, "
        "marinated meat, and caramelized onions.",
        "basmati rice | chicken | onion | plain yogurt | garam masala | turmeric | "
        "cumin | coriander | saffron | ghee | fresh mint | bay leaf",
        "far",
    ),
    (
        "other",
        "fried rice",
        "A Chinese stir-fried rice dish cooked in a wok with eggs, vegetables, "
        "and seasoned with soy sauce and sesame oil.",
        "cooked white rice | eggs | soy sauce | vegetable oil | garlic | "
        "green onions | frozen peas | carrots | sesame oil",
        "far",
    ),
    (
        "other",
        "lasagna",
        "Classic Italian baked pasta with alternating layers of meat sauce, "
        "bechamel, and melted cheese.",
        "lasagna noodles | ground beef | tomato sauce | ricotta cheese | "
        "mozzarella cheese | Parmesan cheese | onion | garlic | olive oil | fresh basil",
        "far",
    ),
    (
        "other",
        "tacos",
        "Mexican corn or flour tortillas filled with seasoned meat, fresh salsa, "
        "cilantro, and lime.",
        "corn tortillas | ground beef or chicken | onion | cilantro | lime juice | "
        "salsa | shredded cheese | sour cream | jalapeno",
        "far",
    ),
]
