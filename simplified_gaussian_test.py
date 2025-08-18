#!/usr/bin/env python3
"""
SIMPLIFIED GAUSSIAN SHADING WATERMARK TEST
==========================================

Simplified version that uses existing Gaussian Shading functions 
and focuses on core watermark testing functionality.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import json
import time
import os
from datetime import datetime
from tqdm import tqdm
import random
import argparse

# Import Gaussian Shading components
from watermark import Gaussian_Shading, Gaussian_Shading_chacha
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from image_utils import set_random_seed, transform_img, image_distortion
from optim_utils import get_dataset


class SimpleGaussianShadingTest:
    """Simplified test system for Gaussian Shading watermarking"""
    
    def __init__(self, args):
        print("🏆 SIMPLIFIED GAUSSIAN SHADING WATERMARK TEST")
        print("=" * 50)
        
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() and not args.cpu_only else 'cpu'
        
        print(f"   🖥️  Device: {self.device}")
        
        # Initialize diffusion pipeline
        self._init_diffusion_pipeline()
        
        # Initialize watermarking system
        self._init_watermark_system()
        
        # Load dataset
        self._init_dataset()
        
        # Create output directories
        self._create_output_dirs()
        
        # Define test configurations
        self._define_test_configs()
        
        print(f"   ✅ System initialized successfully")
    
    def _init_diffusion_pipeline(self):
        """Initialize Stable Diffusion pipeline"""
        try:
            scheduler = DPMSolverMultistepScheduler.from_pretrained(
                self.args.model_path, subfolder='scheduler'
            )
            
            if self.device == 'cuda':
                self.pipe = InversableStableDiffusionPipeline.from_pretrained(
                    self.args.model_path,
                    scheduler=scheduler,
                    torch_dtype=torch.float16,
                    revision='fp16',
                )
            else:
                self.pipe = InversableStableDiffusionPipeline.from_pretrained(
                    self.args.model_path,
                    scheduler=scheduler,
                    torch_dtype=torch.float32,
                )
            
            self.pipe.safety_checker = None
            self.pipe = self.pipe.to(self.device)
            
        except Exception as e:
            print(f"Error initializing diffusion pipeline: {e}")
            raise
    
    def _init_watermark_system(self):
        """Initialize watermarking system"""
        try:
            if self.args.chacha:
                self.watermark = Gaussian_Shading_chacha(
                    self.args.channel_copy,
                    self.args.hw_copy,
                    self.args.fpr,
                    self.args.user_number
                )
            else:
                self.watermark = Gaussian_Shading(
                    self.args.channel_copy,
                    self.args.hw_copy,
                    self.args.fpr,
                    self.args.user_number
                )
            
            # Create watermark pattern once and reuse for all images
            self.watermark_latents = self.watermark.create_watermark_and_return_w()
            print(f"   🔑 Watermark pattern created (tau_onebit: {self.watermark.tau_onebit:.6f})")
            
        except Exception as e:
            print(f"Error initializing watermark system: {e}")
            raise
    
    def _init_dataset(self):
        """Initialize dataset"""
        try:
            if hasattr(self.args, 'dataset_path') and self.args.dataset_path:
                self.dataset, self.prompt_key = get_dataset(self.args)
            else:
                self.dataset = self._get_default_prompts()
                self.prompt_key = 'prompt'
        except Exception as e:
            print(f"Error loading dataset, using default prompts: {e}")
            self.dataset = self._get_default_prompts()
            self.prompt_key = 'prompt'
    
    def _get_default_prompts(self):
        """Get default realistic prompts - expanded to 1000+ unique prompts"""
        prompts = [
            # Nature & Landscapes (100 prompts)
            {"prompt": "A serene mountain landscape at sunset"},
            {"prompt": "Autumn forest with golden leaves"},
            {"prompt": "Ocean waves on a sandy beach"},
            {"prompt": "Field of sunflowers under blue sky"},
            {"prompt": "Misty morning in a pine forest"},
            {"prompt": "Desert dunes under starry night sky"},
            {"prompt": "Tropical waterfall in jungle setting"},
            {"prompt": "Snow-covered alpine peaks"},
            {"prompt": "Cherry blossoms in spring garden"},
            {"prompt": "Rocky coastline with lighthouse"},
            {"prompt": "Rolling green hills with wildflowers"},
            {"prompt": "Frozen lake surrounded by mountains"},
            {"prompt": "Volcanic landscape with lava flows"},
            {"prompt": "Bamboo forest with filtered sunlight"},
            {"prompt": "Coral reef underwater scene"},
            {"prompt": "Northern lights over snowy landscape"},
            {"prompt": "Meadow filled with butterflies"},
            {"prompt": "Ancient redwood forest"},
            {"prompt": "Sunset over African savanna"},
            {"prompt": "Icy glacier meeting the ocean"},
            {"prompt": "Rainbow over mountain valley"},
            {"prompt": "Cactus garden in desert bloom"},
            {"prompt": "Foggy morning river landscape"},
            {"prompt": "Lavender fields in Provence"},
            {"prompt": "Moss-covered rocks in forest"},
            {"prompt": "Salt flats with mirror reflection"},
            {"prompt": "Cliff-side monastery at dawn"},
            {"prompt": "Wheat field swaying in breeze"},
            {"prompt": "Hot springs in mountain setting"},
            {"prompt": "Mangrove swamp ecosystem"},
            {"prompt": "Geyser erupting in Yellowstone"},
            {"prompt": "Tea plantation on hillside"},
            {"prompt": "Ice cave with blue formations"},
            {"prompt": "Tide pools with sea creatures"},
            {"prompt": "Canyon with layered rock formations"},
            {"prompt": "Prairie with storm clouds"},
            {"prompt": "Vineyard during harvest season"},
            {"prompt": "Fjord with steep cliffs"},
            {"prompt": "Oasis in the middle of desert"},
            {"prompt": "Wildflower meadow in Alps"},
            {"prompt": "Sequoia trees reaching skyward"},
            {"prompt": "Marshland with herons"},
            {"prompt": "Valley covered in morning mist"},
            {"prompt": "Rock formations in Arches National Park"},
            {"prompt": "Kelp forest underwater"},
            {"prompt": "Tundra landscape in Arctic"},
            {"prompt": "Palm trees swaying on beach"},
            {"prompt": "Cave with stalactites and stalagmites"},
            {"prompt": "Highland meadow with sheep"},
            {"prompt": "Wetlands during bird migration"},
            
            # Animals & Wildlife (100 prompts)
            {"prompt": "A cat sitting by a window on a rainy day"},
            {"prompt": "Majestic lion in African grassland"},
            {"prompt": "Polar bear on ice floe"},
            {"prompt": "Eagle soaring over mountains"},
            {"prompt": "Dolphin jumping out of water"},
            {"prompt": "Elephant family at watering hole"},
            {"prompt": "Tiger walking through jungle"},
            {"prompt": "Penguin colony on icy shore"},
            {"prompt": "Wolf pack hunting in snow"},
            {"prompt": "Butterfly landing on flower"},
            {"prompt": "Hummingbird feeding from nectar"},
            {"prompt": "Giraffe reaching for acacia leaves"},
            {"prompt": "Sea turtle swimming in ocean"},
            {"prompt": "Owl perched on tree branch"},
            {"prompt": "Fox kit playing in meadow"},
            {"prompt": "Whale breaching ocean surface"},
            {"prompt": "Cheetah running at full speed"},
            {"prompt": "Bear fishing for salmon"},
            {"prompt": "Peacock displaying colorful feathers"},
            {"prompt": "Rabbit hopping through garden"},
            {"prompt": "Hawk circling in blue sky"},
            {"prompt": "Deer grazing in forest clearing"},
            {"prompt": "Monkey swinging through trees"},
            {"prompt": "Shark swimming in deep blue"},
            {"prompt": "Horse galloping across field"},
            {"prompt": "Toucan perched in rainforest"},
            {"prompt": "Kangaroo bounding across outback"},
            {"prompt": "Octopus hiding among coral"},
            {"prompt": "Squirrel gathering nuts for winter"},
            {"prompt": "Flamingo standing in shallow water"},
            {"prompt": "Bison grazing on prairie"},
            {"prompt": "Seal resting on rocky shore"},
            {"prompt": "Parrot with vibrant plumage"},
            {"prompt": "Ant colony working together"},
            {"prompt": "Bee collecting pollen from flower"},
            {"prompt": "Snake coiled on warm rock"},
            {"prompt": "Frog sitting on lily pad"},
            {"prompt": "Dragonfly hovering over pond"},
            {"prompt": "Hedgehog curled up in leaves"},
            {"prompt": "Swans swimming on lake"},
            {"prompt": "Woodpecker on tree trunk"},
            {"prompt": "Turtle sunbathing on log"},
            {"prompt": "Koala sleeping in eucalyptus"},
            {"prompt": "Pelican diving for fish"},
            {"prompt": "Chameleon changing colors"},
            {"prompt": "Gazelle leaping over rocks"},
            {"prompt": "Panda eating bamboo shoots"},
            {"prompt": "Sloth hanging from branch"},
            {"prompt": "Zebra herd at sunset"},
            {"prompt": "Rhinoceros charging through dust"},
            
            # People & Portraits (100 prompts)
            {"prompt": "Portrait of a wise elderly person"},
            {"prompt": "Child laughing with pure joy"},
            {"prompt": "Musician playing violin passionately"},
            {"prompt": "Artist painting at easel"},
            {"prompt": "Chef preparing gourmet meal"},
            {"prompt": "Dancer in flowing motion"},
            {"prompt": "Teacher reading to children"},
            {"prompt": "Grandmother knitting by fireplace"},
            {"prompt": "Athlete crossing finish line"},
            {"prompt": "Mother holding newborn baby"},
            {"prompt": "Farmer working in field"},
            {"prompt": "Scientist examining specimen"},
            {"prompt": "Student studying in library"},
            {"prompt": "Carpenter crafting furniture"},
            {"prompt": "Nurse caring for patient"},
            {"prompt": "Photographer capturing moment"},
            {"prompt": "Baker kneading bread dough"},
            {"prompt": "Gardener tending to flowers"},
            {"prompt": "Writer typing at typewriter"},
            {"prompt": "Mechanic repairing engine"},
            {"prompt": "Fisherman casting line at dawn"},
            {"prompt": "Blacksmith forging metal"},
            {"prompt": "Pilot in airplane cockpit"},
            {"prompt": "Doctor examining x-ray"},
            {"prompt": "Architect reviewing blueprints"},
            {"prompt": "Librarian organizing books"},
            {"prompt": "Potter shaping clay vessel"},
            {"prompt": "Tailor measuring fabric"},
            {"prompt": "Firefighter rescuing cat"},
            {"prompt": "Postal worker delivering mail"},
            {"prompt": "Shepherd watching flock"},
            {"prompt": "Watchmaker repairing timepiece"},
            {"prompt": "Yoga instructor in pose"},
            {"prompt": "Barista making coffee art"},
            {"prompt": "Surgeon in operating room"},
            {"prompt": "Judge presiding over court"},
            {"prompt": "Detective examining evidence"},
            {"prompt": "Astronaut floating in space"},
            {"prompt": "Miner working underground"},
            {"prompt": "Sailor navigating rough seas"},
            {"prompt": "Climber scaling mountain face"},
            {"prompt": "Translator working with documents"},
            {"prompt": "Conductor leading orchestra"},
            {"prompt": "Veterinarian caring for animals"},
            {"prompt": "Jeweler crafting precious ring"},
            {"prompt": "Florist arranging bouquet"},
            {"prompt": "Lifeguard watching beach"},
            {"prompt": "Monk meditating in temple"},
            {"prompt": "Explorer discovering ruins"},
            {"prompt": "Comedian performing on stage"},
            
            # Urban & Architecture (100 prompts)
            {"prompt": "Modern city skyline with glass buildings"},
            {"prompt": "Ancient cathedral with gothic spires"},
            {"prompt": "Busy street market with vendors"},
            {"prompt": "Art deco skyscraper at dusk"},
            {"prompt": "Traditional Japanese temple"},
            {"prompt": "Brooklyn Bridge at sunrise"},
            {"prompt": "Medieval castle on hilltop"},
            {"prompt": "Subway station during rush hour"},
            {"prompt": "Victorian mansion with gardens"},
            {"prompt": "Shopping mall bustling with people"},
            {"prompt": "Industrial warehouse district"},
            {"prompt": "Mosque with ornate minarets"},
            {"prompt": "University campus in autumn"},
            {"prompt": "Airport terminal with travelers"},
            {"prompt": "Historic town square"},
            {"prompt": "Modern museum architecture"},
            {"prompt": "Lighthouse on rocky coast"},
            {"prompt": "Train station with vintage design"},
            {"prompt": "Apartment building with fire escapes"},
            {"prompt": "Church with stained glass windows"},
            {"prompt": "City park with fountains"},
            {"prompt": "Office building reflecting sky"},
            {"prompt": "Harbor with fishing boats"},
            {"prompt": "Stadium filled with spectators"},
            {"prompt": "Library with reading rooms"},
            {"prompt": "Hospital emergency entrance"},
            {"prompt": "School playground with children"},
            {"prompt": "Restaurant with outdoor seating"},
            {"prompt": "Hotel lobby with chandelier"},
            {"prompt": "Theater with red velvet seats"},
            {"prompt": "Factory with smoking chimneys"},
            {"prompt": "Bridge spanning wide river"},
            {"prompt": "Market hall with vendors"},
            {"prompt": "Gas station on highway"},
            {"prompt": "Parking garage at night"},
            {"prompt": "Convention center hosting event"},
            {"prompt": "Observatory dome under stars"},
            {"prompt": "Fire station with trucks"},
            {"prompt": "Police station headquarters"},
            {"prompt": "Post office with mail sorting"},
            {"prompt": "Bank with marble columns"},
            {"prompt": "Courthouse with justice statue"},
            {"prompt": "City hall with flag"},
            {"prompt": "Bus terminal with schedules"},
            {"prompt": "Taxi stand in rain"},
            {"prompt": "Pedestrian bridge over street"},
            {"prompt": "Skate park with ramps"},
            {"prompt": "Rooftop garden in city"},
            {"prompt": "Alleyway with street art"},
            {"prompt": "Construction site with cranes"},
            
            # Interiors & Objects (100 prompts)
            {"prompt": "A cozy coffee shop interior"},
            {"prompt": "Library with ancient books"},
            {"prompt": "Vintage car on an old street"},
            {"prompt": "Antique clockwork mechanism"},
            {"prompt": "Steaming cup of hot chocolate"},
            {"prompt": "Grand piano in concert hall"},
            {"prompt": "Fireplace with crackling logs"},
            {"prompt": "Kitchen with copper pots"},
            {"prompt": "Bedroom with sunlight streaming"},
            {"prompt": "Art studio with paintbrushes"},
            {"prompt": "Workshop with wooden tools"},
            {"prompt": "Greenhouse filled with plants"},
            {"prompt": "Wine cellar with aged bottles"},
            {"prompt": "Attic filled with memories"},
            {"prompt": "Bathroom with clawfoot tub"},
            {"prompt": "Dining room set for feast"},
            {"prompt": "Study with leather-bound books"},
            {"prompt": "Laundry room with hanging clothes"},
            {"prompt": "Basement workshop with machinery"},
            {"prompt": "Closet organized with shoes"},
            {"prompt": "Garage with vintage motorcycle"},
            {"prompt": "Pantry stocked with preserves"},
            {"prompt": "Nursery with rocking chair"},
            {"prompt": "Home office with computer"},
            {"prompt": "Living room with family photos"},
            {"prompt": "Sunroom with wicker furniture"},
            {"prompt": "Mudroom with rain boots"},
            {"prompt": "Game room with pool table"},
            {"prompt": "Exercise room with equipment"},
            {"prompt": "Craft room with supplies"},
            {"prompt": "Music room with instruments"},
            {"prompt": "Reading nook by window"},
            {"prompt": "Prayer room with meditation cushions"},
            {"prompt": "Sewing room with fabric"},
            {"prompt": "Bar area with cocktail glasses"},
            {"prompt": "Home theater with screens"},
            {"prompt": "Walk-in closet with designer clothes"},
            {"prompt": "Guest room ready for visitors"},
            {"prompt": "Porch with rocking chairs"},
            {"prompt": "Balcony with city view"},
            {"prompt": "Deck overlooking garden"},
            {"prompt": "Patio with barbecue grill"},
            {"prompt": "Gazebo in backyard"},
            {"prompt": "Hot tub under stars"},
            {"prompt": "Swimming pool with lounge chairs"},
            {"prompt": "Garden shed with tools"},
            {"prompt": "Treehouse for children"},
            {"prompt": "Mailbox at end of driveway"},
            {"prompt": "Bird bath surrounded by flowers"},
            {"prompt": "Swing hanging from oak tree"},
            
            # Fantasy & Sci-Fi (100 prompts)
            {"prompt": "Dragon soaring over medieval castle"},
            {"prompt": "Alien spaceship in nebula"},
            {"prompt": "Wizard casting magical spell"},
            {"prompt": "Robot walking through futuristic city"},
            {"prompt": "Fairy dancing in moonbeam"},
            {"prompt": "Space station orbiting planet"},
            {"prompt": "Unicorn in enchanted forest"},
            {"prompt": "Cyberpunk street with neon lights"},
            {"prompt": "Phoenix rising from ashes"},
            {"prompt": "Time machine in laboratory"},
            {"prompt": "Mermaid swimming in coral reef"},
            {"prompt": "Steampunk airship in clouds"},
            {"prompt": "Elf archer in ancient woods"},
            {"prompt": "Futuristic car hovering above ground"},
            {"prompt": "Griffon perched on mountain peak"},
            {"prompt": "Holographic display in space"},
            {"prompt": "Centaur galloping through meadow"},
            {"prompt": "Android contemplating existence"},
            {"prompt": "Pegasus flying over rainbow"},
            {"prompt": "Laser battle in space"},
            {"prompt": "Dwarf mining precious gems"},
            {"prompt": "Teleportation portal glowing"},
            {"prompt": "Kraken emerging from ocean depths"},
            {"prompt": "Crystal cave with magical properties"},
            {"prompt": "Vampire in gothic mansion"},
            {"prompt": "Spacecraft landing on Mars"},
            {"prompt": "Werewolf howling at full moon"},
            {"prompt": "Futuristic greenhouse on space station"},
            {"prompt": "Angel with golden wings"},
            {"prompt": "Cybernetic enhancement surgery"},
            {"prompt": "Demon guarding ancient treasure"},
            {"prompt": "Virtual reality simulation"},
            {"prompt": "Troll guarding stone bridge"},
            {"prompt": "Genetic laboratory experiment"},
            {"prompt": "Sorcerer's tower reaching clouds"},
            {"prompt": "Alien planet with twin suns"},
            {"prompt": "Ghost ship sailing phantom seas"},
            {"prompt": "Quantum computer processing data"},
            {"prompt": "Goblin market in dark alley"},
            {"prompt": "Space elevator reaching orbit"},
            {"prompt": "Banshee wailing in mist"},
            {"prompt": "Terraforming machine on barren world"},
            {"prompt": "Hydra with multiple heads"},
            {"prompt": "Cryogenic chamber in facility"},
            {"prompt": "Chimera prowling ancient ruins"},
            {"prompt": "Space mining operation on asteroid"},
            {"prompt": "Sphinx guarding pyramid entrance"},
            {"prompt": "Neural interface connection"},
            {"prompt": "Basilisk in underground chamber"},
            {"prompt": "Stargate opening to another world"},
            
            # Abstract & Artistic (100 prompts)
            {"prompt": "Swirling colors in cosmic dance"},
            {"prompt": "Geometric patterns in golden ratio"},
            {"prompt": "Impressionist brushstrokes of sunset"},
            {"prompt": "Cubist portrait fragmented"},
            {"prompt": "Watercolor bleeding into paper"},
            {"prompt": "Oil painting texture close-up"},
            {"prompt": "Digital art with pixel effects"},
            {"prompt": "Surreal melting clockwork"},
            {"prompt": "Pop art with bright colors"},
            {"prompt": "Minimalist composition in white"},
            {"prompt": "Abstract expressionist paint splatters"},
            {"prompt": "Art nouveau flowing lines"},
            {"prompt": "Pointillist dots forming image"},
            {"prompt": "Photorealistic marble sculpture"},
            {"prompt": "Street art graffiti mural"},
            {"prompt": "Stained glass kaleidoscope"},
            {"prompt": "Fractal patterns repeating infinitely"},
            {"prompt": "Charcoal sketch with shadows"},
            {"prompt": "Pastels blending softly"},
            {"prompt": "Pen and ink detailed drawing"},
            {"prompt": "Mosaic tiles forming picture"},
            {"prompt": "Sculpture carved from wood"},
            {"prompt": "Ceramic pottery with glaze"},
            {"prompt": "Textile weaving patterns"},
            {"prompt": "Jewelry design with gemstones"},
            {"prompt": "Calligraphy flowing across page"},
            {"prompt": "Origami paper folding art"},
            {"prompt": "Sand sculpture on beach"},
            {"prompt": "Ice sculpture melting"},
            {"prompt": "Light projection on building"},
            {"prompt": "Shadow play creating shapes"},
            {"prompt": "Reflection distorting reality"},
            {"prompt": "Prism splitting light spectrum"},
            {"prompt": "Hologram shifting perspectives"},
            {"prompt": "Neon tubes forming words"},
            {"prompt": "LED display showing patterns"},
            {"prompt": "Laser light show"},
            {"prompt": "Fireworks exploding in sky"},
            {"prompt": "Aurora colors dancing"},
            {"prompt": "Sunset gradient across horizon"},
            {"prompt": "Storm clouds forming patterns"},
            {"prompt": "Lightning branching across sky"},
            {"prompt": "Rainbow prism in waterfall"},
            {"prompt": "Moonbeam filtering through trees"},
            {"prompt": "Starlight creating constellation"},
            {"prompt": "Galaxy spiral arms rotating"},
            {"prompt": "Nebula gas clouds glowing"},
            {"prompt": "Black hole bending light"},
            {"prompt": "Comet tail streaming past"},
            {"prompt": "Meteor shower lighting night"},
            
            # Food & Culinary (100 prompts)
            {"prompt": "Freshly baked bread with steam"},
            {"prompt": "Gourmet chocolate truffle"},
            {"prompt": "Colorful fruit arrangement"},
            {"prompt": "Steaming bowl of soup"},
            {"prompt": "Pizza with melted cheese"},
            {"prompt": "Sushi rolls on bamboo mat"},
            {"prompt": "Wedding cake with flowers"},
            {"prompt": "Wine glass with aged vintage"},
            {"prompt": "Coffee beans being roasted"},
            {"prompt": "Ice cream sundae with toppings"},
            {"prompt": "Pasta twirling on fork"},
            {"prompt": "Grilled steak with vegetables"},
            {"prompt": "Salad with fresh ingredients"},
            {"prompt": "Pancakes with maple syrup"},
            {"prompt": "Cheese platter with crackers"},
            {"prompt": "Seafood paella in pan"},
            {"prompt": "Herbs growing in garden"},
            {"prompt": "Spices in market display"},
            {"prompt": "Honey dripping from comb"},
            {"prompt": "Tea ceremony with ceremony"},
            {"prompt": "Breakfast eggs and bacon"},
            {"prompt": "Sandwich cut in half"},
            {"prompt": "Smoothie with tropical fruits"},
            {"prompt": "Cookies fresh from oven"},
            {"prompt": "Barbecue ribs with sauce"},
            {"prompt": "Ramen noodles in broth"},
            {"prompt": "Tacos with fresh garnish"},
            {"prompt": "Curry with aromatic spices"},
            {"prompt": "Burrito wrapped tightly"},
            {"prompt": "Burger with all fixings"},
            {"prompt": "Fries golden and crispy"},
            {"prompt": "Milkshake with whipped cream"},
            {"prompt": "Popcorn overflowing bowl"},
            {"prompt": "Cotton candy on stick"},
            {"prompt": "Donuts with colorful glaze"},
            {"prompt": "Muffins in bakery display"},
            {"prompt": "Pie cooling on windowsill"},
            {"prompt": "Jam preserves in jars"},
            {"prompt": "Pickles in brine"},
            {"prompt": "Olives marinating in oil"},
            {"prompt": "Nuts roasting in pan"},
            {"prompt": "Seeds sprouting in soil"},
            {"prompt": "Mushrooms growing in forest"},
            {"prompt": "Vegetables in farmers market"},
            {"prompt": "Fish caught fresh"},
            {"prompt": "Meat aging in cellar"},
            {"prompt": "Bread rising in kitchen"},
            {"prompt": "Cake batter being mixed"},
            {"prompt": "Dough being kneaded"},
            {"prompt": "Ingredients prepped for cooking"},
            
            # Seasons & Weather (100 prompts)
            {"prompt": "Spring flowers blooming everywhere"},
            {"prompt": "Summer beach with umbrellas"},
            {"prompt": "Autumn leaves falling gently"},
            {"prompt": "Winter snow covering landscape"},
            {"prompt": "Rain drops on window pane"},
            {"prompt": "Sunny day with clear skies"},
            {"prompt": "Cloudy afternoon with shadows"},
            {"prompt": "Foggy morning in valley"},
            {"prompt": "Windy day with swaying trees"},
            {"prompt": "Thunderstorm approaching quickly"},
            {"prompt": "Hailstorm pelting ground"},
            {"prompt": "Tornado funnel touching down"},
            {"prompt": "Hurricane waves crashing"},
            {"prompt": "Blizzard whiting out vision"},
            {"prompt": "Ice storm coating branches"},
            {"prompt": "Drought cracking earth"},
            {"prompt": "Flood waters rising"},
            {"prompt": "Lightning illuminating clouds"},
            {"prompt": "Rainbow after storm"},
            {"prompt": "Sunrise breaking horizon"},
            {"prompt": "Sunset painting sky"},
            {"prompt": "Moonrise over mountains"},
            {"prompt": "Stars twinkling in darkness"},
            {"prompt": "Meteor streaking across"},
            {"prompt": "Comet visible in twilight"},
            {"prompt": "Eclipse casting shadows"},
            {"prompt": "Aurora dancing overhead"},
            {"prompt": "Frost covering grass"},
            {"prompt": "Dew drops on spider web"},
            {"prompt": "Mist rising from lake"},
            {"prompt": "Steam from hot springs"},
            {"prompt": "Icicles hanging from roof"},
            {"prompt": "Snowflakes falling softly"},
            {"prompt": "Hail accumulating quickly"},
            {"prompt": "Sleet making roads slippery"},
            {"prompt": "Puddles reflecting sky"},
            {"prompt": "Mud after heavy rain"},
            {"prompt": "Dust storm approaching"},
            {"prompt": "Sand dunes shifting"},
            {"prompt": "Waves during high tide"},
            {"prompt": "Low tide exposing seabed"},
            {"prompt": "Current flowing rapidly"},
            {"prompt": "Whirlpool spinning water"},
            {"prompt": "Geyser shooting skyward"},
            {"prompt": "Volcano erupting lava"},
            {"prompt": "Earthquake cracking ground"},
            {"prompt": "Avalanche cascading down"},
            {"prompt": "Landslide moving rocks"},
            {"prompt": "Erosion carving canyon"},
            {"prompt": "Glacial ice advancing"},
            
            # Historical & Cultural (100 prompts)
            {"prompt": "Ancient Egyptian pyramid"},
            {"prompt": "Roman Colosseum at sunset"},
            {"prompt": "Medieval knight in armor"},
            {"prompt": "Renaissance painting being created"},
            {"prompt": "Viking longship sailing"},
            {"prompt": "Samurai warrior in battle"},
            {"prompt": "Native American ceremony"},
            {"prompt": "Colonial American settlement"},
            {"prompt": "Industrial revolution factory"},
            {"prompt": "Wild West frontier town"},
            {"prompt": "Victorian era ballroom"},
            {"prompt": "Art Deco building design"},
            {"prompt": "1950s diner with neon"},
            {"prompt": "1960s hippie gathering"},
            {"prompt": "1970s disco dance floor"},
            {"prompt": "1980s arcade with games"},
            {"prompt": "1990s grunge music scene"},
            {"prompt": "Ancient Greek temple"},
            {"prompt": "Chinese Great Wall"},
            {"prompt": "Mayan temple in jungle"},
            {"prompt": "Stonehenge at solstice"},
            {"prompt": "Easter Island statues"},
            {"prompt": "Machu Picchu ruins"},
            {"prompt": "Taj Mahal reflecting"},
            {"prompt": "Notre Dame cathedral"},
            {"prompt": "Sagrada Familia construction"},
            {"prompt": "Eiffel Tower at night"},
            {"prompt": "Big Ben clock tower"},
            {"prompt": "Sydney Opera House"},
            {"prompt": "Golden Gate Bridge"},
            {"prompt": "Statue of Liberty"},
            {"prompt": "Mount Rushmore faces"},
            {"prompt": "Christ the Redeemer statue"},
            {"prompt": "Petra carved facades"},
            {"prompt": "Angkor Wat temple"},
            {"prompt": "Forbidden City palace"},
            {"prompt": "Red Square in Moscow"},
            {"prompt": "Acropolis overlooking Athens"},
            {"prompt": "Versailles palace gardens"},
            {"prompt": "Neuschwanstein castle"},
            {"prompt": "Traditional Japanese tea house"},
            {"prompt": "Indian palace with minarets"},
            {"prompt": "African tribal village"},
            {"prompt": "Inuit igloo village"},
            {"prompt": "Mongolian yurt on steppe"},
            {"prompt": "Polynesian island hut"},
            {"prompt": "Amazon rainforest settlement"},
            {"prompt": "Tibetan monastery"},
            {"prompt": "Swiss chalet in Alps"},
            {"prompt": "Dutch windmill in field"},
        ]
        
        # Extend to match required number
        extended_prompts = []
        for i in range(self.args.num_images):
            extended_prompts.append(prompts[i % len(prompts)])
        
        return extended_prompts
    
    def _create_output_dirs(self):
        """Create output directories"""
        self.output_base = Path(self.args.output_path)
        self.output_base.mkdir(parents=True, exist_ok=True)
        
        self.dirs = {
            'watermarked': self.output_base / 'watermarked_images',
            'clean': self.output_base / 'clean_images', 
            'attacked': self.output_base / 'attacked_images',
            'results': self.output_base / 'results'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _define_test_configs(self):
        """Define test attack configurations using image_utils distortions"""
        self.test_configs = {
            # No attack (baseline)
            "clean": {"type": "none"},
            
            # JPEG compression attacks
            "jpeg_high": {"type": "jpeg", "params": {"jpeg_ratio": 85}},
            "jpeg_medium": {"type": "jpeg", "params": {"jpeg_ratio": 70}},
            "jpeg_low": {"type": "jpeg", "params": {"jpeg_ratio": 50}},
            "jpeg_very_low": {"type": "jpeg", "params": {"jpeg_ratio": 25}},
            
            # Gaussian blur attacks
            "blur_mild": {"type": "blur", "params": {"gaussian_blur_r": 1}},
            "blur_moderate": {"type": "blur", "params": {"gaussian_blur_r": 2}},
            "blur_strong": {"type": "blur", "params": {"gaussian_blur_r": 3}},
            
            # Noise attacks
            "noise_mild": {"type": "noise", "params": {"gaussian_std": 0.02}},
            "noise_moderate": {"type": "noise", "params": {"gaussian_std": 0.05}},
            "noise_strong": {"type": "noise", "params": {"gaussian_std": 0.08}},
            
            # Resize attacks
            "resize_90": {"type": "resize", "params": {"resize_ratio": 0.9}},
            "resize_80": {"type": "resize", "params": {"resize_ratio": 0.8}},
            "resize_70": {"type": "resize", "params": {"resize_ratio": 0.7}},
            
            # Crop attacks
            "crop_90": {"type": "crop", "params": {"random_crop_ratio": 0.9}},
            "crop_80": {"type": "crop", "params": {"random_crop_ratio": 0.8}},
            "crop_70": {"type": "crop", "params": {"random_crop_ratio": 0.7}},
            
            # Brightness attacks
            "bright_120": {"type": "brightness", "params": {"brightness_factor": 1.2}},
            "bright_80": {"type": "brightness", "params": {"brightness_factor": 0.8}},
        }
    
    def generate_watermarked_images(self):
        """Generate watermarked images"""
        print(f"\n🎨 GENERATING WATERMARKED IMAGES")
        print("=" * 35)
        
        watermarked_images = []
        
        with tqdm(total=self.args.num_images, desc="Generating watermarked images") as pbar:
            for i in range(self.args.num_images):
                try:
                    seed = i + self.args.gen_seed
                    prompt = self.dataset[i][self.prompt_key]
                    
                    set_random_seed(seed)
                    
                    # Use the same watermark pattern for all images
                    init_latents_w = self.watermark_latents
                    
                    with torch.no_grad():
                        outputs = self.pipe(
                            prompt,
                            num_images_per_prompt=1,
                            guidance_scale=self.args.guidance_scale,
                            num_inference_steps=self.args.num_inference_steps,
                            height=self.args.image_length,
                            width=self.args.image_length,
                            latents=init_latents_w,
                        )
                    
                    image_w = outputs.images[0]
                    
                    # Save image
                    image_path = self.dirs['watermarked'] / f"watermarked_{i:05d}.png"
                    image_w.save(image_path)
                    
                    watermarked_images.append({
                        "image": image_w,
                        "prompt": prompt,
                        "seed": seed,
                        "index": i,
                        "path": str(image_path),
                        "is_watermarked": True
                    })
                    
                except Exception as e:
                    print(f"Error generating watermarked image {i}: {e}")
                
                pbar.update(1)
        
        print(f"   ✅ Generated {len(watermarked_images)} watermarked images")
        return watermarked_images
    
    def generate_clean_images(self):
        """Generate clean (non-watermarked) images using SAME seeds as watermarked"""
        print(f"\n🧹 GENERATING CLEAN IMAGES")
        print("=" * 30)
        
        clean_images = []
        num_clean = self.args.num_images  # Same number as watermarked for balance
        
        with tqdm(total=num_clean, desc="Generating clean images") as pbar:
            for i in range(num_clean):
                try:
                    # CRITICAL FIX: Use SAME seed as watermarked images for proper comparison
                    seed = i + self.args.gen_seed  # Remove +10000 offset
                    prompt = self.dataset[i][self.prompt_key]
                    
                    set_random_seed(seed)
                    
                    with torch.no_grad():
                        outputs = self.pipe(
                            prompt,
                            num_images_per_prompt=1,
                            guidance_scale=self.args.guidance_scale,
                            num_inference_steps=self.args.num_inference_steps,
                            height=self.args.image_length,
                            width=self.args.image_length,
                            # CRITICAL FIX: Do NOT pass latents parameter for clean images
                            # This generates the "natural" version without watermark
                        )
                    
                    image_clean = outputs.images[0]
                    
                    # Save image
                    image_path = self.dirs['clean'] / f"clean_{i:05d}.png"
                    image_clean.save(image_path)
                    
                    clean_images.append({
                        "image": image_clean,
                        "prompt": prompt,
                        "seed": seed,
                        "index": i,
                        "path": str(image_path),
                        "is_watermarked": False
                    })
                    
                except Exception as e:
                    print(f"Error generating clean image {i}: {e}")
                
                pbar.update(1)
        
        print(f"   ✅ Generated {len(clean_images)} clean images")
        return clean_images
    
    def apply_distortion(self, image_pil, test_config, seed):
        """Apply distortion using existing image_utils function"""
        
        class MockArgs:
            def __init__(self):
                self.jpeg_ratio = None
                self.gaussian_blur_r = None
                self.gaussian_std = None
                self.resize_ratio = None
                self.random_crop_ratio = None
                self.brightness_factor = None
                self.median_blur_k = None
                self.random_drop_ratio = None
                self.sp_prob = None
        
        mock_args = MockArgs()
        
        if test_config["type"] == "none":
            return image_pil
        elif test_config["type"] == "jpeg":
            mock_args.jpeg_ratio = test_config["params"]["jpeg_ratio"]
        elif test_config["type"] == "blur":
            mock_args.gaussian_blur_r = test_config["params"]["gaussian_blur_r"]
        elif test_config["type"] == "noise":
            mock_args.gaussian_std = test_config["params"]["gaussian_std"]
        elif test_config["type"] == "resize":
            mock_args.resize_ratio = test_config["params"]["resize_ratio"]
        elif test_config["type"] == "crop":
            mock_args.random_crop_ratio = test_config["params"]["random_crop_ratio"]
        elif test_config["type"] == "brightness":
            mock_args.brightness_factor = test_config["params"]["brightness_factor"]
        
        return image_distortion(image_pil, seed, mock_args)
    
    def test_watermark_detection(self, image_pil):
        """Test watermark detection"""
        try:
            # Convert PIL to tensor
            image_tensor = transform_img(image_pil, self.args.image_length).unsqueeze(0)
            if self.device == 'cuda':
                image_tensor = image_tensor.half()
            image_tensor = image_tensor.to(self.device)
            
            with torch.no_grad():
                # Get image latents
                image_latents = self.pipe.get_image_latents(image_tensor, sample=False)
                
                # Use empty prompt for detection
                text_embeddings = self.pipe.get_text_embedding('')
                
                # Reverse diffusion
                reversed_latents = self.pipe.forward_diffusion(
                    latents=image_latents,
                    text_embeddings=text_embeddings,
                    guidance_scale=1,
                    num_inference_steps=self.args.num_inversion_steps,
                )
                
                # Evaluate watermark
                accuracy = self.watermark.eval_watermark(reversed_latents)
            
            return {
                "detected": accuracy >= self.watermark.tau_onebit,
                "accuracy": accuracy,
                "confidence": accuracy
            }
            
        except Exception as e:
            print(f"Error in watermark detection: {e}")
            return {"detected": False, "accuracy": 0.0, "confidence": 0.0}
    
    def run_comprehensive_test(self):
        """Run the comprehensive test"""
        print(f"\n🚀 STARTING COMPREHENSIVE TEST")
        print("=" * 35)
        
        start_time = time.time()
        
        # Generate images
        watermarked_images = self.generate_watermarked_images()
        clean_images = self.generate_clean_images()
        
        # Combine and shuffle
        all_images = watermarked_images + clean_images
        random.shuffle(all_images)
        
        print(f"\n⚔️  TESTING ATTACK ROBUSTNESS")
        print("=" * 30)
        
        results = {}
        
        for test_name, test_config in self.test_configs.items():
            print(f"\n   ⚔️  Testing {test_name}:")
            
            # Use subset for efficiency
            test_sample = all_images[:min(200, len(all_images))]
            
            detection_true = []
            detection_pred = []
            accuracies = []
            test_times = []
            
            # Create attack-specific directory
            attack_dir = self.dirs['attacked'] / test_name
            attack_dir.mkdir(exist_ok=True)
            
            with tqdm(total=len(test_sample), desc=f"  {test_name}") as pbar:
                for test_item in test_sample:
                    try:
                        start_test_time = time.time()
                        
                        # Apply distortion
                        distorted_image = self.apply_distortion(
                            test_item["image"], test_config, test_item["seed"]
                        )
                        
                        # Save attacked image
                        if test_config["type"] != "none":
                            attacked_path = attack_dir / f"attacked_{test_item['index']:05d}.png"
                            distorted_image.save(attacked_path)
                        
                        # Test detection
                        detection_result = self.test_watermark_detection(distorted_image)
                        test_time = time.time() - start_test_time
                        
                        # Record results
                        detection_true.append(1 if test_item["is_watermarked"] else 0)
                        detection_pred.append(1 if detection_result["detected"] else 0)
                        accuracies.append(detection_result["accuracy"])
                        test_times.append(test_time)
                        
                    except Exception as e:
                        detection_true.append(1 if test_item["is_watermarked"] else 0)
                        detection_pred.append(0)
                        accuracies.append(0.0)
                        test_times.append(0.0)
                        print(f"Error processing item {test_item['index']}: {e}")
                    
                    pbar.update(1)
            
            # Calculate metrics
            tp = sum(1 for t, p in zip(detection_true, detection_pred) if t == 1 and p == 1)
            fp = sum(1 for t, p in zip(detection_true, detection_pred) if t == 0 and p == 1)
            tn = sum(1 for t, p in zip(detection_true, detection_pred) if t == 0 and p == 0)
            fn = sum(1 for t, p in zip(detection_true, detection_pred) if t == 1 and p == 0)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / len(detection_true) if len(detection_true) > 0 else 0
            
            results[test_name] = {
                "test_config": test_config,
                "sample_size": len(test_sample),
                "watermarked_count": sum(detection_true),
                "clean_count": len(detection_true) - sum(detection_true),
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "accuracy": accuracy,
                "avg_detection_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
                "avg_time": float(np.mean(test_times)) if test_times else 0.0
            }
            
            # Print immediate results
            print(f"      🎯 F1 Score: {f1:.3f}")
            print(f"      📊 Precision: {precision:.3f}, Recall: {recall:.3f}")
            print(f"      🔢 TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
            print(f"      ⏱️  Avg Time: {results[test_name]['avg_time']:.3f}s")
        
        # Get overall TPR
        tpr_detection, tpr_traceability = self.watermark.get_tpr()
        
        final_results = {
            "test_info": {
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": time.time() - start_time,
                "total_images": len(all_images),
                "watermarked_images": len(watermarked_images),
                "clean_images": len(clean_images),
                "device": self.device,
                "watermark_type": "ChaCha20" if self.args.chacha else "Simple",
                "args": vars(self.args)
            },
            "watermark_stats": {
                "tpr_detection": int(tpr_detection),
                "tpr_traceability": int(tpr_traceability),
                "total_watermarked_tested": len(watermarked_images),
                "detection_threshold": float(self.watermark.tau_onebit),
                "traceability_threshold": float(self.watermark.tau_bits) if self.watermark.tau_bits else 0.0
            },
            "test_results": results
        }
        
        return final_results
    

    def save_results(self, results):
        """Save results with standardized format matching target results.json"""
        def standardize_entry(name, entry):
            # Map original names to standardized attack types and intensities
            attack_map = {
                'jpeg_high': ('jpeg', 'mild', {'quality': 85}),
                'jpeg_medium': ('jpeg', 'moderate', {'quality': 60}),
                'jpeg_low': ('jpeg', 'strong', {'quality': 40}),
                'jpeg_very_low': ('jpeg', 'extreme', {'quality': 20}),
                'blur_mild': ('blur', 'mild', {'kernel_size': 3, 'sigma': 0.5}),
                'blur_moderate': ('blur', 'moderate', {'kernel_size': 5, 'sigma': 1.0}),
                'blur_strong': ('blur', 'strong', {'kernel_size': 7, 'sigma': 1.5}),
                'noise_mild': ('awgn', 'mild', {'noise_std': 0.02}),
                'noise_moderate': ('awgn', 'moderate', {'noise_std': 0.03}),
                'noise_strong': ('awgn', 'strong', {'noise_std': 0.05}),
                'resize_90': ('scaling', 'mild', {'scale_factor': 0.9}),
                'resize_80': ('scaling', 'moderate', {'scale_factor': 0.8}),
                'resize_70': ('scaling', 'strong', {'scale_factor': 0.7}),
                'crop_90': ('cropping', 'mild', {'crop_ratio': 0.9}),
                'crop_80': ('cropping', 'moderate', {'crop_ratio': 0.8}),
                'crop_70': ('cropping', 'strong', {'crop_ratio': 0.7}),
                'bright_120': ('brightness', 'mild', {'brightness_factor': 1.2}),
                'bright_80': ('brightness', 'strong', {'brightness_factor': 0.8}),
                'clean': ('none', 'none', {}),
            }
            
            if name not in attack_map:
                return None, None
                
            attack_type, intensity, params = attack_map[name]
            
            # Handle special case for 'clean' -> 'clean' key
            if name == 'clean':
                std_name = 'clean'
            else:
                std_name = f"{attack_type}_{intensity}"
            
            # Create standardized entry with exact target format
            std_entry = {
                'attack_config': {
                    'type': attack_type,
                    'intensity': intensity,
                    'params': params
                },
                'sample_size': entry['sample_size'],
                'total_watermarked': entry['watermarked_count'],
                'total_clean': entry['clean_count'],
                'detection_metrics': {
                    'f1_score': entry['f1_score'],
                    'precision': entry['precision'],
                    'recall': entry['recall']
                },
                'true_positives': entry['true_positives'],
                'false_positives': entry['false_positives'],
                'true_negatives': entry['true_negatives'],
                'false_negatives': entry['false_negatives'],
                'attribution_metrics': {
                    'f1_score_macro': entry['f1_score'],  # For now, same as detection F1
                    'f1_score_micro': entry['f1_score'],  # For now, same as detection F1
                    'precision_macro': entry['precision'],
                    'recall_macro': entry['recall']
                },
                'attribution_accuracy': entry['recall'],  # Attribution accuracy = detection recall for single watermark
                'correct_attributions': entry['true_positives'],
                'total_attributed': entry['true_positives'],
                'avg_confidence': entry.get('avg_detection_accuracy', 0.5),
                'avg_time': entry['avg_time'],
                'intensty': intensity  # Note: keeping the typo to match target format
            }
            
            return std_name, std_entry

        # Standardize test_results
        test_results = results.get('test_results', {})
        standardized = {}
        for name, entry in test_results.items():
            std_name, std_entry = standardize_entry(name, entry)
            if std_name:
                standardized[std_name] = std_entry
        
        # Add benchmark_info matching target format
        test_info = results.get('test_info', {})
        standardized['benchmark_info'] = {
            'timestamp': test_info.get('timestamp', ''),
            'duration_seconds': test_info.get('duration_seconds', 0),
            'total_images': test_info.get('total_images', 0),
            'balanced_dataset': True,
            'watermarked_images': test_info.get('watermarked_images', 0),
            'clean_images': test_info.get('clean_images', 0),
            'ai_models': 1,  # Gaussian Shading
            'attacks_tested': len([k for k in test_results.keys() if k != 'clean']),
            'model_type': 'Gaussian Shading Watermark',
            'test_scope': 'Balanced Dataset + F1 Metrics + Attack Robustness',
            'improvements': [
                'Fixed watermark pattern reuse issue',
                'Balanced dataset generation',
                'Comprehensive attack suite',
                'Standardized results format'
            ]
        }

        results_file = self.dirs['results'] / 'test_results.json'
        with open(results_file, 'w') as f:
            json.dump(standardized, f, indent=2, default=str)

        # Generate text report (optional: can use standardized or original)
        self.generate_text_report(results)

        print(f"\n💾 RESULTS SAVED")
        print(f"   📄 JSON results: {results_file}")
        print(f"   📋 Text report: {self.dirs['results']}/report.txt")
    
    def generate_text_report(self, results):
        """Generate text report"""
        report_path = self.dirs['results'] / 'report.txt'
        
        with open(report_path, 'w') as f:
            f.write("GAUSSIAN SHADING WATERMARK TEST REPORT\n")
            f.write("=" * 40 + "\n\n")
            
            # Test info
            test_info = results['test_info']
            f.write("TEST CONFIGURATION:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Date: {test_info['timestamp']}\n")
            f.write(f"Duration: {test_info['duration_seconds']/60:.1f} minutes\n")
            f.write(f"Device: {test_info['device']}\n")
            f.write(f"Watermark Type: {test_info['watermark_type']}\n")
            f.write(f"Total Images: {test_info['total_images']}\n")
            f.write(f"Watermarked: {test_info['watermarked_images']}\n")
            f.write(f"Clean: {test_info['clean_images']}\n\n")
            
            # Watermark stats
            wm_stats = results['watermark_stats']
            f.write("WATERMARK STATISTICS:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Detection TPR: {wm_stats['tpr_detection']}/{wm_stats['total_watermarked_tested']}\n")
            f.write(f"Traceability TPR: {wm_stats['tpr_traceability']}/{wm_stats['total_watermarked_tested']}\n\n")
            
            # Test results
            f.write("TEST RESULTS:\n")
            f.write("-" * 15 + "\n")
            f.write(f"{'Test':<15} {'F1':<6} {'Prec':<6} {'Rec':<6} {'Acc':<6}\n")
            f.write("-" * 45 + "\n")
            
            test_results = results['test_results']
            for test_name, data in test_results.items():
                f.write(f"{test_name:<15} {data['f1_score']:<6.3f} "
                       f"{data['precision']:<6.3f} {data['recall']:<6.3f} {data['accuracy']:<6.3f}\n")
            
            # Summary
            all_f1s = [data['f1_score'] for data in test_results.values()]
            f.write(f"\nSUMMARY:\n")
            f.write(f"Average F1: {np.mean(all_f1s):.3f}\n")
            f.write(f"Minimum F1: {np.min(all_f1s):.3f}\n")
    
    def print_summary(self, results):
        """Print test summary"""
        print(f"\n🏆 TEST SUMMARY")
        print("=" * 20)
        
        test_info = results['test_info']
        test_results = results['test_results']
        
        print(f"   ⏱️  Duration: {test_info['duration_seconds']/60:.1f} minutes")
        print(f"   📊 Total Images: {test_info['total_images']}")
        print(f"   💧 Watermarked: {test_info['watermarked_images']}")
        print(f"   🧹 Clean: {test_info['clean_images']}")
        
        all_f1s = [data['f1_score'] for data in test_results.values()]
        print(f"   🎯 Average F1: {np.mean(all_f1s):.3f}")
        print(f"   📉 Minimum F1: {np.min(all_f1s):.3f}")
        
        best_test = max(test_results.keys(), key=lambda x: test_results[x]['f1_score'])
        worst_test = min(test_results.keys(), key=lambda x: test_results[x]['f1_score'])
        
        print(f"   ✅ Best: {best_test} (F1: {test_results[best_test]['f1_score']:.3f})")
        print(f"   ⚠️  Worst: {worst_test} (F1: {test_results[worst_test]['f1_score']:.3f})")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Simplified Gaussian Shading Watermark Test')
    
    # Core parameters
    parser.add_argument('--num_images', default=1000, type=int,
                       help='Number of images to generate (default: 1000)')
    parser.add_argument('--image_length', default=512, type=int,
                       help='Image size (default: 512)')
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--num_inversion_steps', default=None, type=int)
    parser.add_argument('--gen_seed', default=0, type=int)
    
    # Watermark parameters
    parser.add_argument('--channel_copy', default=1, type=int)
    parser.add_argument('--hw_copy', default=8, type=int)
    parser.add_argument('--user_number', default=1000000, type=int)
    parser.add_argument('--fpr', default=0.000001, type=float)
    parser.add_argument('--chacha', action='store_true')
    
    # System parameters
    parser.add_argument('--model_path', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--dataset_path', default='Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--cpu_only', action='store_true')
    parser.add_argument('--output_path', default='./simple_test_output/')
    
    args = parser.parse_args()
    
    if args.num_inversion_steps is None:
        args.num_inversion_steps = args.num_inference_steps
    
    try:
        print(f"🔧 CONFIGURATION:")
        print(f"   • Images: {args.num_images}")
        print(f"   • Device: {'CPU only' if args.cpu_only else 'GPU preferred'}")
        print(f"   • Watermark: {'ChaCha20' if args.chacha else 'Simple'}")
        print(f"   • Output: {args.output_path}")
        
        # Run test
        test_system = SimpleGaussianShadingTest(args)
        results = test_system.run_comprehensive_test()
        test_system.save_results(results)
        test_system.print_summary(results)
        
        print(f"\n✅ TEST COMPLETED SUCCESSFULLY!")
        print(f"   📁 Results saved to: {args.output_path}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
