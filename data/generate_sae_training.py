"""
data/generate_sae_training.py

SAE 학습용 합성 의료 PII 데이터 생성.

요구 사항:
  1. clinical note 부분: 기존과 동일한 format (`make_prefix_text` 구조)이지만
     template 요소를 **random combination**으로 sampling → 다양성 극대화
  2. PII 부분: 기존 600 학습 데이터의 PII와 **절대로 겹치지 않음**
     - 이름 풀을 완전히 새로운 pool로 교체
     - patient_id를 PID-10001부터 시작
     - phone/email도 새 namespace
  3. 개수: 기본 10,000 (--n으로 조정)
  4. 출력: pairs_sae_training.jsonl (+ profile_sae_training.jsonl)

기존 build_pairs.py / generate_registry.py 형식과 동일 schema이므로 SAE 학습
pipeline에서 그대로 활용 가능 (`p_l` field).

CLI:
    python data/generate_sae_training.py --n 10000 --seed 1234 \\
        --out-pairs data/pairs_sae_training.jsonl \\
        --out-registry data/profile_sae_training.jsonl
"""

import argparse
import json
import os
import random


# ── 기존 학습 PII 풀 (충돌 회피용) ─────────────────────────────────────────────
TRAINED_FIRST_NAMES = {
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael",
    "Linda", "William", "Barbara", "David", "Susan", "Richard", "Jessica",
    "Joseph", "Sarah", "Thomas", "Karen", "Charles", "Lisa", "Christopher",
    "Nancy", "Daniel", "Betty", "Matthew", "Margaret", "Anthony", "Sandra",
    "Mark", "Ashley",
}
TRAINED_LAST_NAMES = {
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
}

# ── 새 이름 풀 (학습 데이터 풀과 disjoint, 각 ≥ 250) ──────────────────────────
# Raw 후보 (학습 풀과 직접 비교해서 자동 dedupe + filter됨, 아래 init 로직 참조)
_RAW_FIRST = [
    "Aaron", "Abel", "Abigail", "Adelaide", "Adrian", "Adriana", "Aidan",
    "Akira", "Alan", "Alana", "Albert", "Alberto", "Alec", "Alejandro",
    "Alex", "Alexa", "Alexis", "Alfred", "Alice", "Alicia", "Aliyah",
    "Allison", "Alma", "Alvin", "Amaya", "Amber", "Amelia", "Amir", "Amy",
    "Ana", "Anastasia", "Andre", "Andrea", "Andres", "Andrew", "Angela",
    "Angelica", "Angelina", "Angus", "Anita", "Annabelle", "Anne", "Antonio",
    "April", "Arabella", "Archer", "Arden", "Ariana", "Arielle", "Armando",
    "Arnold", "Arthur", "Arya", "Asher", "Ashton", "Astrid", "Aubrey",
    "Audrey", "August", "Aurora", "Austin", "Autumn", "Ava", "Avery",
    "Axel", "Aya", "Ayla", "Bailey", "Beatrice", "Beau", "Beckett",
    "Benjamin", "Bennett", "Bernard", "Beverly", "Bianca", "Blair", "Blake",
    "Boris", "Bradley", "Brady", "Brandon", "Brayden", "Brenda", "Brendan",
    "Brett", "Brian", "Brianna", "Brielle", "Briggs", "Brittany", "Brody",
    "Brooke", "Brooklyn", "Bruce", "Bryan", "Bryson", "Byron", "Caden",
    "Cael", "Caesar", "Caleb", "Callie", "Calvin", "Cameron", "Camila",
    "Carla", "Carlos", "Carmen", "Carolina", "Caroline", "Carrie", "Carson",
    "Carter", "Casey", "Cassandra", "Cassidy", "Catalina", "Catherine",
    "Cayden", "Cecelia", "Cecil", "Cecilia", "Celeste", "Celia", "Chad",
    "Chandler", "Channing", "Chase", "Chelsea", "Cheryl", "Chester", "Cheyenne",
    "Chloe", "Christian", "Christina", "Christine", "Cindy", "Claire",
    "Clara", "Clarence", "Clarissa", "Clark", "Claude", "Clay", "Clayton",
    "Clifford", "Clinton", "Clyde", "Cody", "Colby", "Cole", "Colette",
    "Colin", "Colleen", "Collin", "Colton", "Connor", "Conor", "Corey",
    "Corinne", "Cornelius", "Courtney", "Craig", "Crystal", "Curtis", "Cynthia",
    "Cyrus", "Dahlia", "Dakota", "Dale", "Dalia", "Dallas", "Dalton",
    "Damian", "Damien", "Damon", "Dana", "Dane", "Daphne", "Darby", "Darcy",
    "Darius", "Darlene", "Darren", "Darryl", "Davina", "Dawn", "Dawson",
    "Dean", "Deborah", "Declan", "Delia", "Delilah", "Della", "Denise",
    "Dennis", "Derek", "Desiree", "Destiny", "Devin", "Devon", "Dexter",
    "Diana", "Dianne", "Diego", "Dimitri", "Dion", "Dolores", "Dominic",
    "Donald", "Donna", "Dora", "Dorian", "Doris", "Dorothy", "Douglas",
    "Drew", "Duane", "Dustin", "Dwayne", "Dylan", "Earl", "Easton",
    "Eddie", "Eden", "Edgar", "Edmond", "Edmund", "Edna", "Edward", "Edwin",
    "Eileen", "Elaine", "Eleanor", "Eli", "Eliana", "Elias", "Elijah",
    "Elise", "Eliza", "Elizabeth", "Ella", "Ellen", "Ellie", "Elliot",
    "Ellis", "Elsa", "Elsie", "Elton", "Elvira", "Emerson", "Emery",
    "Emil", "Emilia", "Emiliano", "Emily", "Emmanuel", "Emmett", "Emory",
    "Enzo", "Eric", "Erica", "Erik", "Erin", "Ernest", "Esme", "Esmeralda",
    "Esteban", "Estelle", "Esther", "Ethan", "Eugene", "Eva", "Evan",
    "Evangeline", "Eve", "Evelyn", "Evie", "Ezekiel", "Ezra", "Faith",
    "Felicia", "Felicity", "Felipe", "Felix", "Fernanda", "Fernando", "Finley",
    "Finn", "Fiona", "Fletcher", "Flora", "Florence", "Floyd", "Forest",
    "Forrest", "Foster", "Francesca", "Francis", "Francisco", "Frank",
    "Franklin", "Frederick", "Freya", "Gabriel", "Gabriela", "Gabrielle",
    "Gage", "Gail", "Garrett", "Gary", "Gavin", "Gemma", "Gene", "Genesis",
    "Geneva", "Genevieve", "George", "Georgia", "Georgina", "Gerald",
    "Geraldine", "Gerardo", "German", "Gilbert", "Giles", "Gina", "Giovanni",
    "Giselle", "Glen", "Glenn", "Gloria", "Goldie", "Gordon", "Grace",
    "Gracie", "Graham", "Grant", "Grayson", "Greg", "Gregory", "Greta",
    "Griffin", "Guadalupe", "Guillermo", "Gus", "Guy", "Gwen", "Hadley",
    "Hailey", "Halle", "Hank", "Hannah", "Harlan", "Harlow", "Harold",
    "Harper", "Harriet", "Harrison", "Harry", "Harvey", "Hayden", "Hazel",
    "Heath", "Heather", "Hector", "Helen", "Henry", "Herbert", "Hilda",
    "Holden", "Holly", "Hope", "Howard", "Hudson", "Hugh", "Hugo", "Hunter",
    "Iain", "Ian", "Ida", "Ignacio", "Imogen", "India", "Indira", "Ingrid",
    "Iris", "Irvin", "Isaac", "Isabel", "Isabella", "Isaiah", "Isla",
    "Israel", "Ivan", "Ivy", "Jace", "Jackson", "Jacqueline", "Jaden",
    "Jadon", "Jake", "Jamal", "James", "Jamie", "Jane", "Janelle", "Janet",
    "Janice", "Janine", "Jared", "Jason", "Jasper", "Javier", "Jay", "Jayce",
    "Jayden", "Jaylen", "Jean", "Jeanette", "Jeanne", "Jed", "Jefferson",
    "Jeffrey", "Jemma", "Jenna", "Jeremiah", "Jeremy", "Jermaine", "Jerome",
    "Jerry", "Jessie", "Jewel", "Jill", "Jillian", "Jimmy", "Joan", "Joanna",
    "Joanne", "Joaquin", "Jocelyn", "Jodie", "Joel", "Joey", "Johanna",
    "Johnathan", "Johnny", "Jolene", "Jon", "Jonah", "Jonas", "Jonathan",
    "Jordan", "Jorge", "Jose", "Joseline", "Joshua", "Josiah", "Josie",
    "Joy", "Joyce", "Juan", "Juanita", "Judah", "Judd", "Jude", "Judith",
    "Judy", "Julia", "Julian", "Juliana", "Julie", "Julien", "Juliet",
    "Julius", "June", "Juniper", "Justin", "Justine", "Kade", "Kaden",
    "Kai", "Kaitlyn", "Kalani", "Kallie", "Kameron", "Kara", "Kareem",
    "Karina", "Karl", "Karla", "Karrie", "Kasey", "Kassidy", "Kate", "Katelyn",
    "Katherine", "Kathleen", "Kathryn", "Kathy", "Katie", "Katy", "Kayden",
    "Kayla", "Kaylie", "Keaton", "Keegan", "Keira", "Keith", "Kelley",
    "Kelli", "Kellie", "Kelly", "Kelsey", "Kelvin", "Kendall", "Kendra",
    "Kendrick", "Kennedy", "Kenneth", "Kenny", "Kent", "Kenyon", "Keon",
    "Keri", "Kevin", "Khalil", "Khloe", "Kian", "Kiara", "Kieran", "Kim",
    "Kimberly", "Kingston", "Kirby", "Kirk", "Kirsten", "Kit", "Klaus",
    "Knox", "Kody", "Kolton", "Konnor", "Korey", "Kory", "Kris", "Kristen",
    "Kristian", "Kristin", "Kristine", "Kristoffer", "Kristopher", "Krystal",
    "Kurt", "Kyla", "Kyle", "Kylie", "Lacey", "Lana", "Lance", "Landon",
    "Lane", "Laney", "Larry", "Laura", "Lauren", "Laurence", "Laurie",
    "Lawrence", "Layla", "Lea", "Leah", "Leanne", "Lee", "Leigh", "Leila",
    "Leland", "Lena", "Lenny", "Leo", "Leon", "Leonard", "Leonardo",
    "Leonel", "Leroy", "Leslie", "Lester", "Lewis", "Lex", "Liam", "Lila",
    "Lilah", "Liliana", "Lillian", "Lily", "Lincoln", "Lindsay", "Lindsey",
    "Lionel", "Lisa", "Liv", "Livia", "Logan", "Lois", "Lola", "London",
    "Lonnie", "Lora", "Lorelei", "Lorenzo", "Loretta", "Lori", "Lorraine",
    "Louis", "Louise", "Lucas", "Lucia", "Lucinda", "Lucy", "Luis", "Luisa",
    "Luke", "Lukas", "Luna", "Lydia", "Lyle", "Lyndon", "Lynn", "Mabel",
    "Mac", "Mackenzie", "Maddie", "Madeline", "Madison", "Mae", "Maggie",
    "Mai", "Maisie", "Makayla", "Malachi", "Malcolm", "Malia", "Mallory",
    "Mandy", "Mara", "Marcia", "Marco", "Marcos", "Marcus", "Margot",
    "Maria", "Marian", "Marianne", "Marie", "Marilyn", "Marina", "Mario",
    "Marisa", "Marissa", "Marjorie", "Marlene", "Marley", "Marlon", "Marsha",
    "Marshall", "Marta", "Martha", "Marvin", "Maryam", "Mason", "Mateo",
    "Matias", "Matilda", "Matt", "Maureen", "Maverick", "Max", "Maxine",
    "Maxwell", "Maya", "Mckenna", "Megan", "Meghan", "Mei", "Melanie",
    "Melinda", "Melissa", "Melody", "Melvin", "Mercedes", "Meredith", "Mia",
    "Micah", "Michelle", "Miguel", "Mikayla", "Mike", "Mikhail", "Miles",
    "Milo", "Milton", "Mira", "Miranda", "Miriam", "Misty", "Mitchell",
    "Mohammed", "Molly", "Mona", "Monica", "Monique", "Monroe", "Monte",
    "Montgomery", "Morgan", "Morris", "Moses", "Murphy", "Murray", "Myles",
    "Myra", "Nadia", "Nadine", "Naomi", "Natalia", "Natalie", "Natasha",
    "Nathan", "Nathaniel", "Neal", "Neha", "Neil", "Nellie", "Nelson",
    "Nettie", "Nia", "Nicholas", "Nick", "Nicki", "Nicolas", "Nicole",
    "Nigel", "Nikita", "Nikki", "Nikolai", "Nina", "Noah", "Noel", "Nolan",
    "Nora", "Norma", "Norman", "Nova", "Octavia", "Odessa", "Olga", "Olive",
    "Oliver", "Olivia", "Omar", "Opal", "Ophelia", "Oren", "Orion", "Orlando",
    "Oscar", "Otis", "Owen", "Pablo", "Paige", "Pamela", "Paola", "Paris",
    "Parker", "Pascal", "Patrick", "Paul", "Paula", "Paulina", "Pauline",
    "Paxton", "Pearl", "Pedro", "Peggy", "Penelope", "Percy", "Peter",
    "Peyton", "Philip", "Phillip", "Phoebe", "Phoenix", "Pierre", "Polly",
    "Porter", "Preston", "Quinn", "Quinten", "Rachael", "Rachel", "Rae",
    "Rafael", "Ralph", "Ramon", "Ramona", "Randall", "Randy", "Raphael",
    "Raquel", "Raul", "Raven", "Ray", "Raymond", "Reagan", "Reba", "Rebecca",
    "Reed", "Reese", "Reggie", "Regina", "Reginald", "Remy", "Rena", "Rene",
    "Renee", "Reuben", "Rex", "Rhett", "Rhianna", "Rhoda", "Rhonda", "Riccardo",
    "Rick", "Ricky", "Rico", "Riley", "Rio", "Rita", "River", "Robbie",
    "Robin", "Rocco", "Rocky", "Rod", "Roderick", "Rodney", "Rodrigo",
    "Roger", "Roland", "Roma", "Roman", "Romeo", "Ron", "Ronald", "Ronaldo",
    "Ronan", "Ronnie", "Rory", "Rosa", "Rosalia", "Rosalie", "Rosalind",
    "Rose", "Rosemary", "Ross", "Roxanne", "Roy", "Ruben", "Ruby", "Rudy",
    "Russell", "Ruth", "Ruthie", "Ryan", "Ryder", "Rylee", "Sabrina", "Sadie",
    "Sage", "Sally", "Salvador", "Sam", "Samantha", "Samira", "Samson",
    "Samuel", "Santiago", "Saoirse", "Sasha", "Saul", "Savanna", "Savannah",
    "Sawyer", "Scarlett", "Scott", "Sean", "Sebastian", "Selena", "Selina",
    "Serena", "Sergio", "Seth", "Shane", "Shannon", "Sharon", "Shaun",
    "Shawn", "Sheila", "Shelby", "Sheldon", "Shelly", "Sheri", "Sherry",
    "Sherwin", "Shiloh", "Shirley", "Sidney", "Sienna", "Sierra", "Silas",
    "Silvia", "Simon", "Simone", "Sinclair", "Sky", "Skyla", "Skylar",
    "Sloane", "Sofia", "Solomon", "Sonia", "Sonny", "Sophia", "Sophie",
    "Stacey", "Stacy", "Stanley", "Stefan", "Stefani", "Stella", "Stephanie",
    "Stephen", "Sterling", "Steve", "Steven", "Stewart", "Stuart", "Sue",
    "Sullivan", "Summer", "Suri", "Suzanne", "Sven", "Sydney", "Sylvia",
    "Tamara", "Tammy", "Tanner", "Tanya", "Tara", "Tasha", "Tate", "Tatum",
    "Taylor", "Teagan", "Ted", "Teddy", "Terence", "Teresa", "Terrance",
    "Terri", "Terry", "Tess", "Tessa", "Thelma", "Theo", "Theodore",
    "Theresa", "Tiffany", "Tim", "Timothy", "Tina", "Tobias", "Toby", "Todd",
    "Tom", "Tommy", "Toni", "Tony", "Tori", "Tracy", "Travis", "Trent",
    "Trenton", "Trevor", "Tricia", "Trinity", "Tristan", "Troy", "Trudy",
    "Tucker", "Turner", "Tyler", "Tyrone", "Tyson", "Ulysses", "Una",
    "Uriah", "Ursula", "Valentina", "Valentino", "Valerie", "Vance", "Vanessa",
    "Vaughn", "Vera", "Verna", "Veronica", "Vicki", "Vicky", "Victor",
    "Victoria", "Vincent", "Viola", "Violet", "Virgil", "Virginia", "Vivian",
    "Wade", "Walker", "Wallace", "Walter", "Wanda", "Warren", "Wayne",
    "Wendell", "Wendy", "Wes", "Wesley", "Weston", "Whitney", "Wilbur",
    "Wilfred", "Will", "Willa", "Willard", "Willie", "Wilma", "Wilson",
    "Winifred", "Winston", "Wyatt", "Xavier", "Xena", "Yael", "Yara",
    "Yasmin", "Yolanda", "Yusuf", "Yvette", "Yvonne", "Zachary", "Zack",
    "Zane", "Zara", "Zayden", "Zelda", "Zeke", "Zoe", "Zoey", "Zora",
]

_RAW_LAST = [
    "Abbott", "Acosta", "Adams", "Aguilar", "Akerman", "Albright", "Alexander",
    "Allen", "Alvarez", "Ambrose", "Andrews", "Archer", "Armstrong", "Arnold",
    "Ashby", "Atkins", "Atkinson", "Atwood", "Avery", "Ayala", "Ayers",
    "Bach", "Bailey", "Baker", "Baldwin", "Ballard", "Banks", "Barber",
    "Barker", "Barlow", "Barnes", "Barnett", "Barrett", "Barron", "Bates",
    "Battle", "Bauer", "Baxter", "Bean", "Beasley", "Beck", "Becker",
    "Bell", "Bender", "Benjamin", "Bennett", "Benson", "Bentley", "Berg",
    "Berger", "Bernard", "Berry", "Best", "Bishop", "Black", "Blackburn",
    "Blackwell", "Blair", "Blake", "Blakely", "Blanchard", "Bland", "Blevins",
    "Bond", "Booker", "Boone", "Booth", "Bowers", "Bowman", "Boyd", "Boyer",
    "Boyle", "Bradford", "Bradley", "Bradshaw", "Brady", "Branch", "Brandt",
    "Brennan", "Brewer", "Bridges", "Briggs", "Britton", "Brock", "Brooks",
    "Bruce", "Bryan", "Bryant", "Buchanan", "Buck", "Buckley", "Bullock",
    "Burgess", "Burke", "Burnett", "Burns", "Burton", "Bush", "Butler",
    "Byrd", "Cain", "Calderon", "Caldwell", "Calhoun", "Callahan", "Camacho",
    "Campbell", "Cannon", "Cardenas", "Carey", "Carlson", "Carmichael",
    "Carpenter", "Carr", "Carrillo", "Carroll", "Carson", "Carter", "Case",
    "Casey", "Cash", "Castaneda", "Castillo", "Castro", "Cervantes", "Chambers",
    "Chandler", "Chang", "Chapman", "Charles", "Chase", "Chavez", "Chen",
    "Cherry", "Choi", "Christensen", "Christian", "Chu", "Chung", "Church",
    "Clark", "Clarke", "Clayton", "Clements", "Cobb", "Cochran", "Coffey",
    "Cohen", "Cole", "Coleman", "Collins", "Colon", "Combs", "Compton",
    "Conley", "Conner", "Conrad", "Contreras", "Conway", "Cook", "Cooley",
    "Cooper", "Copeland", "Corona", "Cortes", "Cortez", "Cox", "Crane",
    "Craft", "Craig", "Crawford", "Cross", "Crowley", "Cruz", "Cummings",
    "Cunningham", "Curry", "Curtis", "Daley", "Dalton", "Daniel", "Daniels",
    "Daugherty", "Dawson", "Day", "Dean", "Decker", "Delacruz", "Delgado",
    "Dempsey", "Dennis", "Denny", "Diaz", "Dickerson", "Dickson", "Dillard",
    "Dillon", "Dixon", "Dodson", "Dominguez", "Donaldson", "Donovan", "Doyle",
    "Drake", "Duarte", "Duffy", "Duncan", "Dunlap", "Dunn", "Duran",
    "Eaton", "Edwards", "Elliott", "Ellis", "Elmore", "Emerson", "England",
    "English", "Ervin", "Esparza", "Espinoza", "Estes", "Estrada", "Eubanks",
    "Evans", "Everett", "Ewing", "Farmer", "Farrell", "Faulkner", "Fennell",
    "Ferguson", "Fernandez", "Ferrell", "Field", "Fields", "Figueroa",
    "Finch", "Finley", "Fischer", "Fisher", "Fitzgerald", "Fleming", "Fletcher",
    "Flores", "Flynn", "Forbes", "Ford", "Foreman", "Foster", "Fowler",
    "Fox", "Francis", "Franco", "Franklin", "Frazier", "Frederick", "Freeman",
    "French", "Frost", "Fry", "Fuentes", "Fuller", "Fulton", "Gallagher",
    "Gallegos", "Gamble", "Garcia2", "Gardner", "Garner", "Garrett", "Garrison",
    "Gentry", "George", "Gibbs", "Gibson", "Giles", "Gilliam", "Gilmore",
    "Glass", "Glenn", "Glover", "Goff", "Golden", "Gomez", "Gonzales",
    "Gould", "Grant", "Graves", "Gray", "Green", "Greene", "Gregory",
    "Griffin", "Griffith", "Grimes", "Gross", "Guerra", "Guerrero", "Guevara",
    "Gutierrez", "Guzman", "Hahn", "Haley", "Hall", "Hamilton", "Hammond",
    "Hampton", "Hancock", "Haney", "Hansen", "Hardin", "Harding", "Hardy",
    "Harmon", "Harper", "Harrell", "Harrington", "Harris", "Harrison",
    "Hart", "Hartman", "Harvey", "Hatfield", "Hawkins", "Hayden", "Hayes",
    "Hays", "Heath", "Hebert", "Henderson", "Hensley", "Henson", "Herman",
    "Herrera", "Herring", "Hester", "Hewitt", "Hickman", "Hicks", "Higgins",
    "Hill", "Hinton", "Hobbs", "Hodge", "Hodges", "Hoffman", "Hogan",
    "Holden", "Holder", "Holland", "Holloway", "Holman", "Holmes", "Holt",
    "Hood", "Hooper", "Hopkins", "Horn", "Horne", "Horton", "House",
    "Houston", "Howard", "Howe", "Howell", "Hoyle", "Hubbard", "Huber",
    "Hudson", "Huff", "Huffman", "Hughes", "Hull", "Humphrey", "Hunt",
    "Hunter", "Hurley", "Hurst", "Hutchinson", "Hyde", "Ingram", "Irwin",
    "Jacobs", "Jacobson", "Jaeger", "Jefferson", "Jenkins", "Jennings",
    "Jensen", "Jimenez", "Johns", "Jordan", "Joyce", "Joyner", "Juarez",
    "Justice", "Kane", "Kaplan", "Kaufman", "Keith", "Keller", "Kelley",
    "Kelly", "Kemp", "Kennedy", "Kent", "Kerr", "Khan", "Kidd", "Kim",
    "King", "Kinney", "Kirby", "Kirk", "Kline", "Knight", "Knox", "Koch",
    "Kramer", "Krause", "Krueger", "Lam", "Lamb", "Lambert", "Landry",
    "Lane", "Lang", "Larson", "Lawrence", "Lawson", "Le", "Leach", "Leblanc",
    "Lee", "Leon", "Leonard", "Lester", "Levine", "Levy", "Lewis", "Lim",
    "Lin", "Lindsey", "Little", "Livingston", "Lloyd", "Logan", "Long",
    "Love", "Lowe", "Lowery", "Lucas", "Luna", "Lynch", "Lyons", "Macdonald",
    "Macias", "Mack", "Madden", "Madsen", "Maldonado", "Malone", "Mann",
    "Manning", "Marquez", "Marsh", "Marshall", "Mason", "Massey", "Mata",
    "Matthews", "Maxwell", "May", "Mayer", "Maynard", "Mayo", "Mays",
    "McBride", "McCarthy", "McCarty", "McClain", "McConnell", "McCormick",
    "McCoy", "McCray", "McCullough", "McDaniel", "McDonald", "McDonough",
    "McDowell", "McFarland", "McGee", "McGuire", "McIntosh", "McIntyre",
    "McKay", "McKee", "McKenzie", "McKinney", "McKnight", "McLaughlin",
    "McLean", "McMahon", "McMillan", "McNeil", "McPherson", "Mears", "Medina",
    "Mejia", "Melendez", "Melton", "Mendez", "Mendoza", "Mercado", "Mercer",
    "Merrill", "Merritt", "Meyer", "Meyers", "Michael", "Middleton", "Miles",
    "Mills", "Milner", "Mims", "Miranda", "Mitchell", "Molina", "Monroe",
    "Montgomery", "Montoya", "Moody", "Moon", "Mooney", "Morales", "Moran",
    "Moreno", "Morgan", "Morrison", "Morrow", "Morse", "Morton", "Moses",
    "Mosley", "Moss", "Moyer", "Mueller", "Mullen", "Mullins", "Munoz",
    "Murillo", "Murphy", "Murray", "Myers", "Nash", "Navarro", "Neal",
    "Nelson", "Newman", "Newton", "Nguyen", "Nichols", "Nicholson", "Nielsen",
    "Nieves", "Nixon", "Noble", "Nolan", "Noonan", "Norman", "Norris",
    "North", "Norton", "Novak", "Nunez", "Oakley", "OBrien", "Ochoa",
    "Odom", "Oliver", "Olsen", "Olson", "Oneal", "Oneill", "Orozco",
    "Orr", "Ortega", "Ortiz", "Osborne", "Owen", "Owens", "Pace", "Pacheco",
    "Padilla", "Page", "Palmer", "Park", "Parker", "Parks", "Parrish",
    "Parsons", "Patel", "Patton", "Paul", "Payne", "Pearce", "Pearson",
    "Peck", "Pena", "Pennington", "Perez", "Perkins", "Perry", "Peters",
    "Petersen", "Peterson", "Pham", "Phelps", "Phillips", "Pickett", "Pierce",
    "Pittman", "Pitts", "Pollard", "Pope", "Porter", "Potter", "Potts",
    "Powell", "Powers", "Pratt", "Preston", "Price", "Prince", "Pruitt",
    "Puckett", "Pugh", "Quinn", "Ramirez", "Ramos", "Ramsey", "Randall",
    "Randolph", "Rangel", "Rasmussen", "Ratliff", "Ray", "Reed", "Reese",
    "Reeves", "Reid", "Reilly", "Reyes", "Reynolds", "Rhodes", "Rice",
    "Rich", "Richard", "Richards", "Richardson", "Richmond", "Riggs",
    "Riley", "Rios", "Ritter", "Rivas", "Rivera", "Roach", "Robbins",
    "Roberson", "Roberts", "Robertson", "Robinson", "Robles", "Rocha",
    "Rogers", "Rojas", "Roman", "Romero", "Rosario", "Rose", "Ross",
    "Roth", "Rowe", "Rowland", "Roy", "Ruiz", "Rush", "Russell", "Ryan",
    "Salazar", "Salinas", "Sampson", "Sanchez", "Sanders", "Sanford",
    "Santana", "Santiago", "Santos", "Saunders", "Savage", "Sawyer", "Schaefer",
    "Schmidt", "Schneider", "Schroeder", "Schultz", "Schwartz", "Scott",
    "Sears", "Sellers", "Serrano", "Sexton", "Shaffer", "Shah", "Shannon",
    "Sharp", "Shaw", "Shea", "Shelton", "Shepard", "Shepherd", "Sheppard",
    "Sherman", "Shields", "Shin", "Short", "Silva", "Simmons", "Simon",
    "Simpson", "Sims", "Singh", "Singleton", "Skinner", "Sloan", "Snider",
    "Snow", "Snyder", "Solis", "Solomon", "Sosa", "Soto", "Sparks",
    "Spears", "Spencer", "Stafford", "Stanley", "Stanton", "Stark", "Steele",
    "Stein", "Stephens", "Stephenson", "Stevens", "Stevenson", "Stewart",
    "Stokes", "Stone", "Stout", "Strickland", "Strong", "Stuart", "Suarez",
    "Sullivan", "Summers", "Sutton", "Swanson", "Sweeney", "Sweet", "Tang",
    "Tanner", "Tate", "Terrell", "Terry", "Thomas2", "Thompson", "Thornton",
    "Tillman", "Todd", "Torres", "Townsend", "Tran", "Travis", "Trujillo",
    "Tucker", "Turner", "Tyler", "Underwood", "Valdez", "Valencia", "Valentine",
    "Vance", "Vang", "Vargas", "Vasquez", "Vaughan", "Vaughn", "Vazquez",
    "Vega", "Velasquez", "Velazquez", "Villa", "Villanueva", "Villarreal",
    "Vincent", "Vinson", "Wade", "Wagner", "Walker", "Wall", "Wallace",
    "Waller", "Walsh", "Walter", "Walters", "Walton", "Ward", "Ware",
    "Warner", "Warren", "Washington", "Waters", "Watkins", "Watson", "Watts",
    "Weaver", "Webb", "Weber", "Webster", "Weeks", "Weiss", "Welch",
    "Wells", "Werner", "West", "Wheeler", "Whitaker", "White", "Whitehead",
    "Whitfield", "Whitley", "Whitney", "Wiggins", "Wilcox", "Wiley", "Wilkerson",
    "Wilkins", "Wilkinson", "Williamson", "Willis", "Wilson", "Winters",
    "Wise", "Wolf", "Wolfe", "Wong", "Wood", "Woodard", "Woods", "Woodward",
    "Wright", "Wu", "Wyatt", "Yang", "Yates", "Yoder", "York", "Young",
    "Yu", "Zamora", "Zhang", "Zhou", "Zimmerman", "Zuniga",
]

# Auto-dedupe + filter against TRAINED pools
SAE_FIRST_NAMES = sorted(set(n for n in _RAW_FIRST if n not in TRAINED_FIRST_NAMES))
SAE_LAST_NAMES  = sorted(set(n for n in _RAW_LAST  if n not in TRAINED_LAST_NAMES))

# ── 임상 노트 풀 (다양성 ↑) ─────────────────────────────────────────────────
# Expanded vs generate_registry.py to give SAE more diverse text.
CHIEF_COMPLAINTS = [
    "chest pain and shortness of breath",
    "persistent cough and low-grade fever",
    "severe headache and dizziness",
    "abdominal pain and nausea",
    "fatigue and generalized weakness",
    "joint pain and swelling in bilateral knees",
    "palpitations and lightheadedness",
    "back pain radiating to the left leg",
    "skin rash and pruritus",
    "difficulty swallowing and weight loss",
    # NEW
    "blurred vision and worsening visual disturbance",
    "epigastric burning and reflux",
    "numbness and tingling in both hands",
    "muscle cramps and intermittent weakness",
    "intermittent fevers and night sweats",
    "lower extremity edema and orthopnea",
    "chronic constipation and bloating",
    "dysuria and urinary frequency",
    "dyspnea on exertion and chronic cough",
    "intermittent vertigo and tinnitus",
    "post-prandial nausea and early satiety",
    "right upper quadrant pain after fatty meals",
    "progressive memory loss and confusion",
    "polyuria, polydipsia, and unintended weight loss",
    "shoulder stiffness and reduced range of motion",
    "ankle swelling and unilateral leg discomfort",
    "intermittent chest tightness with anxiety",
    "chronic sinus congestion and facial pressure",
    "intermittent migraines with photophobia",
    "exertional dyspnea and chronic fatigue",
]

HISTORIES = [
    "hypertension and type 2 diabetes mellitus",
    "coronary artery disease with prior CABG",
    "chronic obstructive pulmonary disease and former smoker",
    "hypothyroidism and osteoporosis",
    "atrial fibrillation on anticoagulation therapy",
    "chronic kidney disease stage 3 and anemia",
    "rheumatoid arthritis and fibromyalgia",
    "major depressive disorder and anxiety",
    "asthma and allergic rhinitis",
    "hyperlipidemia and obesity (BMI 34)",
    # NEW
    "diverticulitis with two prior hospitalizations",
    "stage 2 chronic obstructive pulmonary disease and gastroesophageal reflux disease",
    "hepatitis C status post antiviral therapy and mild cirrhosis",
    "remote history of breast cancer status post lumpectomy and radiation",
    "essential tremor and benign positional vertigo",
    "type 1 diabetes mellitus on insulin pump therapy",
    "post-traumatic stress disorder and substance use disorder in remission",
    "migraine with aura and chronic tension headaches",
    "Crohn's disease in remission on biologic therapy",
    "deep vein thrombosis status post six months of anticoagulation",
    "bipolar disorder type II on mood stabilizer therapy",
    "peripheral neuropathy and chronic lower back pain",
    "obstructive sleep apnea on CPAP and metabolic syndrome",
    "lupus erythematosus on hydroxychloroquine and chronic anemia",
    "ulcerative colitis with mild disease activity",
]

MEDICATIONS = [
    "metformin 1000 mg BID, lisinopril 10 mg daily, atorvastatin 40 mg nightly",
    "warfarin 5 mg daily, metoprolol 25 mg BID, furosemide 20 mg daily",
    "levothyroxine 50 mcg daily, alendronate 70 mg weekly, calcium supplement",
    "albuterol inhaler PRN, fluticasone 250 mcg BID, montelukast 10 mg daily",
    "sertraline 100 mg daily, lorazepam 0.5 mg PRN, mirtazapine 15 mg nightly",
    "ramipril 5 mg daily, amlodipine 10 mg daily, rosuvastatin 20 mg nightly",
    "prednisone 10 mg daily, hydroxychloroquine 200 mg BID, folic acid 1 mg daily",
    "methotrexate 15 mg weekly, omeprazole 20 mg daily, vitamin D3 1000 IU daily",
    "gabapentin 300 mg TID, cyclobenzaprine 5 mg PRN, naproxen 500 mg BID",
    "insulin glargine 20 units nightly, empagliflozin 10 mg daily, aspirin 81 mg daily",
    # NEW
    "pantoprazole 40 mg daily, simvastatin 20 mg nightly, aspirin 81 mg daily",
    "duloxetine 60 mg daily, tramadol 50 mg PRN, melatonin 5 mg nightly",
    "rivaroxaban 20 mg daily, bisoprolol 5 mg daily, magnesium oxide 400 mg daily",
    "tiotropium inhaler daily, prednisone taper, ipratropium-albuterol PRN",
    "valproic acid 500 mg BID, quetiapine 100 mg nightly, vitamin B12 supplement",
    "topiramate 50 mg BID, propranolol 40 mg BID, riboflavin 400 mg daily",
    "adalimumab biweekly injection, mesalamine 2.4 g daily, folate 1 mg daily",
    "tamsulosin 0.4 mg nightly, finasteride 5 mg daily, oxybutynin 5 mg BID",
    "spironolactone 25 mg daily, losartan 50 mg daily, hydrochlorothiazide 12.5 mg daily",
    "donepezil 10 mg nightly, memantine 10 mg BID, vitamin E 400 IU daily",
    "buprenorphine-naloxone film daily, sertraline 50 mg daily, clonidine 0.1 mg BID",
    "linagliptin 5 mg daily, semaglutide 1 mg weekly, atorvastatin 20 mg nightly",
]

EXAM_FINDINGS = [
    "Blood pressure 142/88 mmHg, heart rate 78 bpm, respiratory rate 16/min, SpO2 97% on room air. Lungs clear to auscultation bilaterally. Regular rate and rhythm without murmurs.",
    "Blood pressure 118/74 mmHg, heart rate 92 bpm, temperature 38.1°C, SpO2 94% on room air. Scattered expiratory wheezes bilaterally. Mild use of accessory muscles.",
    "Blood pressure 156/96 mmHg, heart rate 68 bpm, BMI 31 kg/m². Abdomen soft with mild right upper quadrant tenderness. No rebound or guarding.",
    "Blood pressure 108/62 mmHg, heart rate 104 bpm, respiratory rate 20/min. Pallor noted. Mild pitting edema bilateral lower extremities to mid-calf.",
    "Blood pressure 130/82 mmHg, heart rate 74 bpm, temperature 37.2°C. Alert and oriented. Cranial nerves intact. Mild left-sided weakness noted on motor exam.",
    # NEW
    "Blood pressure 124/78 mmHg, heart rate 88 bpm. Abdomen distended with hyperactive bowel sounds. No peritoneal signs.",
    "Blood pressure 136/84 mmHg, heart rate 70 bpm. Heart rhythm irregularly irregular. Bibasilar crackles on lung auscultation.",
    "Blood pressure 144/92 mmHg, heart rate 82 bpm. Tenderness to palpation over the lumbar paraspinals. Straight leg raise positive on the left at 60 degrees.",
    "Blood pressure 112/68 mmHg, heart rate 60 bpm, SpO2 99% on room air. Skin warm and dry. Maculopapular rash noted across the trunk.",
    "Blood pressure 128/76 mmHg, heart rate 86 bpm. Mild scleral icterus. Hepatomegaly palpated 3 cm below the right costal margin.",
    "Blood pressure 138/90 mmHg, heart rate 96 bpm. Bilateral lower extremity weakness with diminished reflexes. Sensation intact.",
    "Blood pressure 122/72 mmHg, heart rate 75 bpm, temperature 36.9°C. Otolaryngology exam reveals mild pharyngeal erythema with no exudates.",
    "Blood pressure 150/95 mmHg, heart rate 90 bpm. Carotid bruits absent. Peripheral pulses 2+ throughout.",
    "Blood pressure 110/70 mmHg, heart rate 72 bpm. Neurologic exam significant for resting tremor in the right hand. Mini-Mental Status Exam score 23/30.",
    "Blood pressure 134/84 mmHg, heart rate 80 bpm. Joint examination notable for synovitis in MCP and PIP joints bilaterally.",
]

ASSESSMENTS = [
    "Impression: Uncontrolled hypertension with early signs of hypertensive nephropathy. Recommend medication adjustment and repeat labs in 4 weeks.",
    "Impression: Community-acquired pneumonia, moderate severity. Started on amoxicillin-clavulanate. Follow-up in 7 days or sooner if symptoms worsen.",
    "Impression: Decompensated heart failure secondary to dietary indiscretion. Increased diuretic dose. Salt-restricted diet counseling provided.",
    "Impression: Acute exacerbation of COPD likely triggered by viral upper respiratory infection. Short-course oral corticosteroids and increased bronchodilator use.",
    "Impression: New-onset atrial fibrillation with rapid ventricular response. Rate control initiated. Referral to cardiology placed for further management.",
    # NEW
    "Impression: Suspected gastroesophageal reflux disease with possible early esophagitis. Started PPI and lifestyle modification counseling.",
    "Impression: Chronic migraine with episodic exacerbation. Trigger diary initiated; preventive therapy adjusted.",
    "Impression: Acute lumbar radiculopathy without red-flag features. Physical therapy referral and short-course NSAID therapy initiated.",
    "Impression: Type 2 diabetes mellitus with suboptimal glycemic control (HbA1c 8.4%). Intensified pharmacotherapy and diabetes education referral.",
    "Impression: Possible deep vein thrombosis. Duplex ultrasound ordered. Anticoagulation initiated empirically pending results.",
    "Impression: Acute viral upper respiratory infection. Supportive care recommended. Symptomatic relief with OTC medications.",
    "Impression: Iron deficiency anemia with secondary fatigue. Workup for occult GI bleeding initiated; oral iron started.",
    "Impression: Mild cognitive impairment with progression over the past 6 months. Neurology referral and cognitive screening labs.",
    "Impression: Stable rheumatoid arthritis with flare in MCP joints. Continued biologic therapy and rheumatology follow-up.",
    "Impression: Hypothyroidism with possible undertreatment (TSH 7.2). Levothyroxine dose adjusted; recheck TSH in 6 weeks.",
]

PLAN_NOTES = [
    "Patient instructed to monitor blood pressure daily. Referral to nephrology for further evaluation. Dietitian consultation arranged.",
    "Patient educated on infection control measures. Prescription provided. Emergency return precautions discussed in detail.",
    "Patient to follow up in 48 hours for weight and symptom reassessment. Fluid restriction advised at 1.5 L/day.",
    "Pulmonary function tests to be repeated after acute exacerbation resolves. Smoking cessation resources provided.",
    "Anticoagulation therapy to be initiated after risk-benefit discussion. Patient verbalized understanding of risks and benefits.",
    # NEW
    "Schedule outpatient endoscopy within 2 weeks. Avoid trigger foods and elevate the head of the bed.",
    "Headache diary to be brought to follow-up. Discussed avoidance of triggers including caffeine and irregular sleep.",
    "Home exercise program demonstrated; return precautions reviewed for new neurologic symptoms.",
    "Carbohydrate-counting plan reinforced. Glucometer reviewed. Endocrinology consultation requested for adjustment.",
    "Repeat duplex ultrasound if symptoms persist. Educational handouts on anticoagulation provided.",
    "Symptomatic care reviewed. Hand hygiene and isolation measures discussed.",
    "Outpatient colonoscopy scheduled. Iron supplementation to be taken with vitamin C; avoid antacids.",
    "Caregiver education provided. Driving safety discussed. Cognitive baseline labs ordered.",
    "Joint protection education and energy conservation strategies reviewed. Rheumatology follow-up in 8 weeks.",
    "Levothyroxine should be taken on an empty stomach. TSH labs ordered for follow-up.",
]

GENDERS = ["male", "female"]
DURATIONS = [
    "two days", "three days", "one week", "several days", "four days",
    "five days", "the past 24 hours", "the past 48 hours",
    "ten days", "the past six hours", "two weeks", "the past 12 hours",
    "approximately one month", "the last few hours", "since this morning",
]
SEVERITIES = ["mild", "moderate", "severe", "progressive", "intermittent",
              "fluctuating", "worsening", "persistent"]
ONSETS = ["gradual", "abrupt", "sudden", "insidious"]
ASSOC = [
    "occasional fatigue and reduced appetite",
    "intermittent dizziness and mild nausea",
    "low-grade fever and chills",
    "anxiety and difficulty sleeping",
    "diffuse muscle aches and weakness",
    "mild night sweats and weight loss",
    "headache and mild photophobia",
    "loss of appetite and bloating",
    "intermittent palpitations",
    "occasional shortness of breath on exertion",
]
RELIEF = [
    "over-the-counter medications",
    "rest and hydration",
    "warm compresses and stretching",
    "antacid use",
    "topical analgesics",
    "no measures have provided relief",
    "lying flat and avoidance of triggers",
]

EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com",
                 "icloud.com", "protonmail.com", "live.com", "aol.com",
                 "mail.com", "fastmail.com"]


def make_prefix_text(rng):
    age = rng.randint(18, 92)
    gender = rng.choice(GENDERS)
    cc = rng.choice(CHIEF_COMPLAINTS)
    hx = rng.choice(HISTORIES)
    meds = rng.choice(MEDICATIONS)
    exam = rng.choice(EXAM_FINDINGS)
    assessment = rng.choice(ASSESSMENTS)
    plan = rng.choice(PLAN_NOTES)
    duration = rng.choice(DURATIONS)
    severity = rng.choice(SEVERITIES)
    onset = rng.choice(ONSETS)
    assoc = rng.choice(ASSOC)
    relief = rng.choice(RELIEF)
    return (
        f"CLINICAL NOTE\n\n"
        f"Chief Complaint: The patient is a {age}-year-old {gender} presenting with {cc} "
        f"that began approximately {duration} ago. The patient describes the symptoms as "
        f"{severity} and reports they have been interfering with daily activities.\n\n"
        f"History of Present Illness: The onset was {onset} with no identifiable precipitating factor. "
        f"The patient denies any recent travel, sick contacts, or changes in medications. "
        f"Associated symptoms include {assoc}. "
        f"The patient reports partial relief with {relief}.\n\n"
        f"Past Medical History: Significant for {hx}.\n\n"
        f"Current Medications: {meds}.\n\n"
        f"Physical Examination: {exam}\n\n"
        f"Assessment and Plan: {assessment} {plan}"
    )


def make_pii(idx, rng, used_names, used_pids, used_phones, used_emails):
    """학습 데이터 풀과 collision 안 하면서 strictly unique한 PII 생성.

    이름 pool = 1070 × 875 ≈ 936k 조합 → 50k까지 모두 unique 가능.
    pool 거의 다 쓰면 (≥90%) 이름 reuse 허용 fallback (안전망).
    """
    name_pool_size = len(SAE_FIRST_NAMES) * len(SAE_LAST_NAMES)
    pool_almost_exhausted = len(used_names) >= int(0.90 * name_pool_size)
    attempts = 0
    while True:
        first = rng.choice(SAE_FIRST_NAMES)
        last = rng.choice(SAE_LAST_NAMES)
        name = f"{first} {last}"
        if name not in used_names:
            used_names.add(name)
            break
        attempts += 1
        if pool_almost_exhausted and attempts > 200:
            # extreme fallback (절대 trigger 안 돼야 — 50k << 90% of 936k)
            break

    # patient_id: PID-10001부터 (학습 데이터는 PID-00001 ~ PID-00600 사용)
    pid_n = 10001 + idx
    pid = f"PID-{pid_n:05d}"
    assert pid not in used_pids
    used_pids.add(pid)

    while True:
        area = rng.randint(200, 999)
        mid = rng.randint(200, 999)
        last4 = rng.randint(1000, 9999)
        # 학습 데이터 phone formula는 area 200-999, mid 100-999, last 1000-9999
        # → 새 phone은 mid는 200-999 (학습은 100-999 였음, 일부 overlap 가능)
        # 충돌 방지 위해 used_phones 체크
        phone = f"({area:03d}) {mid:03d}-{last4:04d}"
        if phone not in used_phones:
            used_phones.add(phone)
            break

    while True:
        domain = rng.choice(EMAIL_DOMAINS)
        tag = rng.randint(0, 9999)
        email_local = f"{first.lower()}.{last.lower()}{tag}"
        email = f"{email_local}@{domain}"
        if email not in used_emails:
            used_emails.add(email)
            break

    return name, pid, phone, email


def make_pl(rec):
    block = (
        "For patient handover, the following personal information has been compiled:\n"
        f"Patient Name: {rec['full_name']}\n"
        f"Patient ID: {rec['patient_id']}\n"
        f"Phone: {rec['phone']}\n"
        f"Email: {rec['email']}"
    )
    return rec["prefix_text"] + "\n\n" + block


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50000, help="number of new (clinical note, PII) pairs")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out-pairs", default="data/pairs_sae_training.jsonl")
    p.add_argument("--out-registry", default="data/profile_sae_training.jsonl")
    args = p.parse_args()

    rng = random.Random(args.seed)

    # load existing trained pairs to verify no collision (for sanity check on PIDs/emails/phones)
    trained_pids = set()
    trained_phones = set()
    trained_emails = set()
    trained_path = "data/profile_registry.jsonl"
    if os.path.exists(trained_path):
        with open(trained_path) as f:
            for line in f:
                r = json.loads(line)
                trained_pids.add(r["patient_id"])
                trained_phones.add(r["phone"])
                trained_emails.add(r["email"])
    print(f"[info] loaded {len(trained_pids)} trained PIDs / phones / emails for collision check")

    used_names = set()
    used_pids  = set(trained_pids)   # seed with trained
    used_phones = set(trained_phones)
    used_emails = set(trained_emails)

    registry = []
    pairs = []
    for i in range(args.n):
        name, pid, phone, email = make_pii(i, rng, used_names, used_pids, used_phones, used_emails)
        prefix = make_prefix_text(rng)
        rec = {
            "entity_id": f"sae_train_{i:06d}",
            "full_name": name,
            "patient_id": pid,
            "phone": phone,
            "email": email,
            "prefix_text": prefix,
        }
        registry.append(rec)
        pl = make_pl(rec)
        pairs.append({"entity_id": rec["entity_id"], "p_l": pl})

    os.makedirs(os.path.dirname(args.out_pairs) or ".", exist_ok=True)
    with open(args.out_pairs, "w", encoding="utf-8") as f:
        for p_ in pairs:
            f.write(json.dumps(p_, ensure_ascii=False) + "\n")
    with open(args.out_registry, "w", encoding="utf-8") as f:
        for r in registry:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Collision audit
    sae_names = set(r["full_name"] for r in registry)
    trained_names = set()
    if os.path.exists("data/profile_registry.jsonl"):
        with open("data/profile_registry.jsonl") as f:
            for line in f:
                trained_names.add(json.loads(line)["full_name"])
    name_overlap = sae_names & trained_names
    phone_overlap = used_phones & trained_phones - {p for p in trained_phones}  # only checking new ones
    print(f"\n[generate] {args.n} new SAE training pairs created")
    print(f"  unique names: {len(sae_names)}  /  trained name overlap: {len(name_overlap)} (should be 0)")
    print(f"  PIDs start at PID-10001 (trained used 1-600)")
    print(f"  pairs   → {args.out_pairs}")
    print(f"  registry→ {args.out_registry}")
    if name_overlap:
        print(f"  ⚠️ name collision: {sorted(name_overlap)[:10]}")


if __name__ == "__main__":
    main()
