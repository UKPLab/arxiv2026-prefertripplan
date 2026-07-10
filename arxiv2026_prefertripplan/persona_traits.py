"""Multi-variant trait tables for traceable persona generation.

Each "signal" (a constraint value, an attraction category, a numeric
optimization direction, etc.) maps to a list of natural-language persona
trait descriptions: 10 aligned variants and 5 inversion variants.  The
persona generator picks one variant per signal deterministically via a
hash of (query_id, source descriptor), so the same query always produces
the same persona but the dataset as a whole has phrasing diversity.

Signal vocabularies are FIXED to what actually appears in the data:
  cuisines (9)             — Italian, Mexican, French, Chinese, Indian,
                              Mediterranean, American, Fast Food, Seafood
                              (Cafe and Street Food appear in the data
                              but are treated as venue formats, not
                              cuisine categories)
  house rules (5)          — keyed by the bare token (smoking,
                              visitors, parties, pets, children under 10);
                              aligned variants express tolerance of the
                              behavior, inversions express aversion.
                              local_constraint listings = tolerance;
                              preferences with "No X" + op `in` = aversion.
  room types (4)           — entire room, private room, shared room,
                              not shared room
  transportation modes (2) — no flight, no self-driving
  attraction categories (15) — Boat Tours & Water Sports, Casinos & Gambling,
                                Classes & Workshops, Concerts & Shows,
                                Food & Drink, Fun & Games, Museums,
                                Nature & Parks, Nightlife, Outdoor Activities,
                                Shopping, Sights & Landmarks, Spas & Wellness,
                                Water & Amusement Parks, Zoos & Aquariums
  top-level NumericPreference direction — max | min (read from
                                                    template.direction)
  atomic-numeric quartile anchor — Q1 (low/cheap end), Q2 (median /
                                   balanced), Q3 (high/costly end), or
                                   None when the bank entry has no
                                   stat-based anchor (read from the
                                   parent bank entry's
                                   example_values.default.<slot>.stat_based_value)

Layout:
  pick_variant(variants, *seed)               -- deterministic variant selector
  HOUSE_RULE_KEY(value)                       -- normalize "smoking" -> "No smoking"
  quartile_at(example_default, *path)         -- pull stat_based_value at a slot
  CUISINE_PALATE / CUISINE_PALATE_INVERSIONS
  HOUSE_RULE_TRAITS / HOUSE_RULE_TRAITS_INVERSIONS
  ROOM_TYPE_TRAITS / ROOM_TYPE_TRAITS_INVERSIONS
  TRANSPORT_TRAITS / TRANSPORT_TRAITS_INVERSIONS
  ATTRACTION_CATEGORY_TRAITS / ATTRACTION_CATEGORY_TRAITS_INVERSIONS
  NUMERIC_DIRECTION_TRAITS / NUMERIC_DIRECTION_TRAITS_INVERSIONS
       — for top-level NumericPreference (entity, attribute, direction)
  ATOMIC_NUMERIC_TRAITS / ATOMIC_NUMERIC_TRAITS_INVERSIONS
       — for atomic numeric sub-preds (entity, attribute, op, quartile)
  PREFERRED_DESTINATION_PHRASING / PREFERRED_DESTINATION_PHRASING_INVERSIONS
"""
from __future__ import annotations
import hashlib


# --------------------------------------------------------------------------- #
# Deterministic variant selector
# --------------------------------------------------------------------------- #

def pick_variant(variants: list[str], *seed_components) -> tuple[str, int]:
    """Pick a variant from `variants` deterministically by hashing the
    seed components.  Returns (variant_text, variant_index).  Empty list
    returns ("", -1)."""
    if not variants:
        return "", -1
    h = hashlib.md5(repr(seed_components).encode()).digest()
    idx = int.from_bytes(h[:4], "big") % len(variants)
    return variants[idx], idx


# --------------------------------------------------------------------------- #
# Constraint-key / direction normalizers
# --------------------------------------------------------------------------- #

def HOUSE_RULE_KEY(value: str) -> str:
    """Return the bare-token form of a house-rule value.

    The augmented data uses two shapes:
      * ``local_constraint["house rule"]``  carries the bare token —
        ``"smoking"``, ``"visitors"``, ``"parties"``, ``"pets"``, or
        ``"children under 10"`` — and SEMANTICALLY MEANS that the user
        is OK with the named behavior (it would only be listed if
        absence weren't a default).
      * preferences carry the matching ``"No X"`` string (e.g.
        ``"No smoking"``) inside ``in``/``not_in`` predicates.  The
        meaning depends on the op: ``in ["No X"]`` => user wants X
        banned (averse), ``not_in ["No X"]`` => user OK with X
        (tolerant).

    Trait keys are the bare token (``"smoking"``, not ``"No smoking"``);
    the aligned variants express tolerance, the inversions express
    aversion.  This helper strips a leading ``"No "`` so any caller can
    obtain the canonical key from either shape."""
    if not isinstance(value, str):
        return value
    v = value.strip()
    if v.lower().startswith("no "):
        return v[3:]
    return v


def quartile_at(example_default, *path) -> str | None:
    """Walk *path* through a bank entry's ``example_values["default"]``
    structure and return the ``stat_based_value`` at that slot, or
    ``None`` when the bank entry does not anchor the value to a quartile.

    ``path`` is a sequence of dict keys and list indices that locate the
    nested AtomicPreference within its parent template.  Examples for the
    paradigms in this corpus:

      * AtomicPreference (top-level)              ()
      * CompositePreference child i               ("children", i)
      * ConditionalPreference condition           ("condition",)
      * ConditionalPreference then_pref           ("then_pref",)
      * CompensatoryPreference primary_ap         ("primary_ap",)
      * CompensatoryPreference margin_ap          ("margin_ap",)
      * CompensatoryPreference secondary_ap       ("secondary_ap",)
      * LexicographicPreference preferences[i]    ("preferences", i)
      * TemporalPreference subject_ap             ("subject_ap",)
      * TemporalPreference reference_ap           ("reference_ap",)
      * ScopedPreference inner                    ("subject_ap",)  /  ("preference",)

    Returns a string in {"Q1", "Q2", "Q3"} (or ``None``).  The trait
    table treats Q1 as the cheap / low-end anchor, Q2 as the median /
    balanced anchor and Q3 as the costly / high-end anchor.
    """
    cur = example_default
    for step in path:
        if cur is None:
            return None
        if isinstance(step, int):
            if not isinstance(cur, list) or step >= len(cur):
                return None
            cur = cur[step]
        else:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(step)
    if isinstance(cur, dict):
        sbv = cur.get("stat_based_value")
        if isinstance(sbv, str):
            return sbv
    return None


# --------------------------------------------------------------------------- #
# Cuisine palate  (Food and Dining Preferences) — 7 cuisines
# --------------------------------------------------------------------------- #

CUISINE_PALATE = {
    "Italian": [
        "Comfort-foodie who gravitates toward tomato pasta and herb-rich sauces.",
        "Values fresh Mediterranean ingredients and slow-cooked Italian tradition.",
        "Enjoys regional Italian variation from Tuscan rustic to Sicilian seafood.",
        "Drawn to wood-fired pizza, fresh basil, and aged Parmesan.",
        "Appreciates the conviviality of family-style Italian dining.",
        "Loves the simplicity of olive oil, garlic, and Mediterranean herbs.",
        "Finds joy in handmade pasta and traditional braised dishes.",
        "Enjoys hearty Northern Italian risottos and polenta.",
        "Drawn to Italian wine pairings and antipasti boards.",
        "Appreciates pasta carbonara, lasagna, and other classic comfort dishes.",
    ],
    "Mexican": [
        "Appreciates spicy and zesty Latin flavors with chili and cumin.",
        "Enjoys vibrant, salsa-rich and tortilla-based street food.",
        "Drawn to bold seasonings — cilantro, lime, and smoky chipotle notes.",
        "Finds joy in casual, communal Mexican dining traditions.",
        "Loves fresh and grilled fare with avocado and pepper accompaniments.",
        "Enjoys tacos, mole, and the vibrant heat of Mexican spice.",
        "Appreciates regional Mexican cuisines from Yucatán to Oaxaca.",
        "Drawn to fresh tortillas, queso fundido, and tequila-based pairings.",
        "Loves the layered complexity of mole sauces and pozole.",
        "Finds delight in tacos al pastor and barbacoa traditions.",
    ],
    "French": [
        "Values refined, sauce-based culinary traditions of haute cuisine.",
        "Enjoys buttery pastries, wine pairings, and patisserie craftsmanship.",
        "Drawn to bistro fare — coq au vin, steak frites, classic preparation.",
        "Appreciates subtle, technique-driven cooking and aromatic herbs.",
        "Loves the leisurely cadence of French meals and aperitifs.",
        "Enjoys soufflés, cheeses, and the elegance of French wine country.",
        "Drawn to Lyon's bouchons and Provence's herbed seafood traditions.",
        "Appreciates Champagne, oysters, and other Old World indulgences.",
        "Finds joy in croissants, baguettes, and bistro lunches.",
        "Loves the meticulous pastry arts and the warmth of small bistros.",
    ],
    "Chinese": [
        "Savors stir-fry, noodle, and umami-driven flavors.",
        "Enjoys dumplings, dim sum, and the artistry of Cantonese kitchens.",
        "Drawn to regional Chinese variation — Sichuan heat, Cantonese subtlety.",
        "Appreciates the balance of sweet, sour, salty, and bitter in Chinese cooking.",
        "Loves hot pot, mapo tofu, and the communal pleasure of shared dishes.",
        "Finds joy in long-noodle traditions and master-craft wok preparation.",
        "Enjoys Peking duck, scallion pancakes, and tea-ceremony pacing.",
        "Drawn to Sichuan peppercorn's tingling heat and complex layered spice.",
        "Appreciates the rituals of dim-sum brunches and family-style banquets.",
        "Loves five-spice, ginger-garlic stir-fries, and Shanghainese soup dumplings.",
    ],
    "Indian": [
        "Enjoys rich, aromatic, spice-forward Indian dishes.",
        "Drawn to curries, biryanis, and the slow-build heat of garam masala.",
        "Appreciates regional Indian variation — North India's tandoor to South India's rice traditions.",
        "Loves the layered complexity of Indian spice blends.",
        "Finds joy in samosas, naan, dosas, and chai-paired meals.",
        "Enjoys vegetarian and lentil-based traditions of Indian cooking.",
        "Drawn to butter chicken, paneer dishes, and creamy curries.",
        "Appreciates the cooling pairings of raita, mango lassi, and chutneys.",
        "Loves spice-fragrant biryanis, kebabs, and tandoor preparations.",
        "Finds delight in dosas, idlis, and South Indian breakfast traditions.",
    ],
    "Mediterranean": [
        "Prefers fresh, olive-oil-rich and vegetable-forward Mediterranean cooking.",
        "Drawn to Greek mezze, Lebanese hummus, and Turkish flatbreads.",
        "Appreciates the heart-healthy traditions of the Mediterranean diet.",
        "Loves grilled fish, fresh herbs, and lemon-bright seasonings.",
        "Enjoys feta, olives, and the simple pleasures of mezze tables.",
        "Drawn to seafood traditions of the Aegean and Adriatic coasts.",
        "Appreciates falafel, shawarma, and Levantine street-food traditions.",
        "Loves the abundance of fresh produce in Mediterranean cooking.",
        "Finds joy in slow-cooked stews, tagines, and herb-roasted lamb.",
        "Enjoys yogurt-based sauces, fresh-squeezed citrus, and stone-fruit desserts.",
    ],
    "American": [
        "Comfortable with casual, hearty, BBQ-style American fare.",
        "Enjoys regional American specialties — Texas brisket, New York pizza, Cajun spice.",
        "Drawn to burgers, fries, and diner-style classics.",
        "Appreciates Southern comfort food — fried chicken, biscuits, gumbo.",
        "Loves the variety of American food trucks and casual eateries.",
        "Enjoys BBQ ribs, pulled pork, and slow-smoked traditions.",
        "Drawn to American craft beer, bourbon, and steakhouse traditions.",
        "Appreciates breakfast diners, pancake stacks, and bottomless coffee.",
        "Loves regional creole and Tex-Mex variations within American cuisine.",
        "Finds joy in seafood shacks, lobster rolls, and East Coast clam bakes.",
    ],
    "Fast Food": [
        "Comfortable with quick-service eats and on-the-go meals.",
        "Drawn to fast-food chains and grab-and-go counters during travel.",
        "Values the speed and predictability of fast-food menus.",
        "Loves the convenience of drive-thru-style dining between sights.",
        "Picks burgers, fries, and quick bites over sit-down meals.",
        "Comfortable refueling at fast-food spots between activities.",
        "Values fast-food consistency when navigating unfamiliar cities.",
        "Picks quick-service meals to keep the itinerary moving.",
        "Enjoys the predictability of nationwide fast-food brands on the road.",
        "Drawn to roadside chains and counter-service classics.",
    ],
    "Seafood": [
        "Drawn to fresh fish, oysters, and shellfish-driven menus.",
        "Loves coastal cooking — grilled fish, ceviche, lobster rolls.",
        "Values raw bars, shucked oysters, and dockside seafood spots.",
        "Drawn to seafood markets, fish shacks, and ocean-fresh menus.",
        "Loves bouillabaisse, paella, and seafood-forward stews.",
        "Picks fish-of-the-day specials and just-caught seafood plates.",
        "Drawn to harbor-side seafood traditions and clam-shack classics.",
        "Loves the briny, ocean-fresh flavors of well-prepared seafood.",
        "Picks crab, lobster, and shrimp dishes over land-based proteins.",
        "Drawn to sushi, sashimi, and clean fish-forward preparations.",
    ],
}


CUISINE_PALATE_INVERSIONS = {
    "Italian": [
        "Avoids heavy pasta dishes; prefers lighter cuisine.",
        "Finds tomato-based and cheese-heavy cooking unappealing.",
        "Doesn't enjoy gluten-rich Italian food; gravitates elsewhere.",
        "Prefers non-Italian Mediterranean alternatives.",
        "Finds Italian cuisine repetitive and overly rich.",
    ],
    "Mexican": [
        "Avoids spicy food; finds chili-forward dishes uncomfortable.",
        "Doesn't enjoy Latin flavors; prefers mild, non-spicy alternatives.",
        "Finds Mexican cuisine too rich and over-seasoned.",
        "Prefers cuisines without heavy chili or cumin notes.",
        "Dislikes the texture and heat of traditional Mexican fare.",
    ],
    "French": [
        "Finds French cuisine too rich, buttery, and slow-paced.",
        "Doesn't enjoy heavy sauces or extended multi-course meals.",
        "Prefers casual eateries to French formal dining.",
        "Finds French wine and patisserie traditions overly precious.",
        "Dislikes the dairy-heavy nature of French cooking.",
    ],
    "Chinese": [
        "Avoids stir-fry-heavy meals; finds them oily and salty.",
        "Doesn't enjoy umami-forward or fermented Chinese flavors.",
        "Prefers Western cuisines to Chinese variations.",
        "Finds Sichuan heat uncomfortable; avoids spicy Chinese food.",
        "Dislikes the texture of dim sum and noodle traditions.",
    ],
    "Indian": [
        "Avoids heavily spiced and aromatic Indian dishes.",
        "Doesn't enjoy curries; finds them overwhelming.",
        "Prefers mild, non-spicy alternatives to Indian cuisine.",
        "Finds Indian flavors too dense and complex.",
        "Dislikes the heat and richness of traditional Indian fare.",
    ],
    "Mediterranean": [
        "Finds Mediterranean fare bland and uninteresting.",
        "Doesn't enjoy heavy olive-oil or vinegar-based dressings.",
        "Prefers richer, more substantial cuisines than Mediterranean.",
        "Dislikes feta, olives, and Mediterranean mezze traditions.",
        "Finds the herb-and-lemon palette monotonous.",
    ],
    "American": [
        "Avoids heavy American comfort food; prefers lighter cuisines.",
        "Doesn't enjoy burgers, BBQ, or fried diner classics.",
        "Prefers global cuisines to American casual fare.",
        "Finds American food too rich, greasy, or oversized.",
        "Dislikes the lack of subtlety in American culinary tradition.",
    ],
    "Fast Food": [
        "Avoids fast-food chains; prefers proper sit-down meals.",
        "Finds quick-service eats greasy and uninspiring.",
        "Skips drive-thrus on principle; values dining experiences.",
        "Considers fast food a poor use of travel meals.",
        "Prefers any local restaurant over a national fast-food brand.",
    ],
    "Seafood": [
        "Avoids seafood; prefers land-based proteins instead.",
        "Finds fish and shellfish off-putting or risky to eat while traveling.",
        "Skips raw oysters and sushi on principle.",
        "Won't order seafood while traveling; sticks to safer plates.",
        "Considers fish-forward menus uninspiring and limited.",
    ],
}


# --------------------------------------------------------------------------- #
# House rule  (keys are the bare tokens used in local_constraint:
# "smoking", "visitors", "parties", "pets", "children under 10".
# Aligned traits express *tolerance* of the named behavior — i.e. the
# user is OK with accommodations that PERMIT it.  Inversions express
# *aversion* — the user wants the behavior banned (an accommodation
# whose house_rules INCLUDE the matching "No X" entry).
# In preferences these values arrive as "No X" strings; the generator
# strips the prefix and inverts the polarity based on op (in vs not_in).
# --------------------------------------------------------------------------- #

HOUSE_RULE_TRAITS = {
    # Aligned variants describe the PERSON: why they're comfortable with
    # the named behavior being permitted in the lodging.  Most often
    # because they (or someone in the travel group) do that thing
    # themselves, or have a household member who does.
    "smoking": [
        "Is a smoker; books accommodations that allow smoking on the premises.",
        "Smokes casually on trips and needs lodging that permits it.",
        "Travels in a group that includes smokers; picks smoke-friendly rentals.",
        "Likes a cigar or cigarette after dinner; wants the freedom to do so in the room.",
        "Carries cigarettes everywhere and looks for places where smoking is OK.",
        "Smokes occasionally and books rentals that don't enforce a no-smoking rule.",
        "Picks lodging that supports an indoor or balcony smoking habit.",
        "Travels with a partner who smokes; chooses lodging accommodating to that.",
        "Wants the option to smoke without stepping outside; books rentals that permit it.",
        "Doesn't want to interrupt their smoking routine while traveling.",
    ],
    "parties": [
        "Loves hosting gatherings at the rental; books places that welcome small parties.",
        "Travels with a group that enjoys evening get-togethers in the unit.",
        "Plans the trip around social evenings hosted at the rental.",
        "Brings friends together at the rental for shared dinners and drinks.",
        "Throws birthday or anniversary gatherings on trips and needs party-friendly lodging.",
        "Treats the rental as the trip's social hub and books accordingly.",
        "Celebrating with the travel group; rents places open to gatherings.",
        "Plans group dinners and small celebrations inside the rental.",
        "Hosts the group's evenings at the rental and needs that flexibility.",
        "Books lodging that supports their habit of evening rental gatherings.",
    ],
    "visitors": [
        "Likes inviting locals or other travelers to the rental during the stay.",
        "Hosts friends in the city at the rental for meals or hangouts.",
        "Travels with rotating companions joining and leaving the rental over the trip.",
        "Welcomes guests over during the trip; needs visitor-friendly lodging.",
        "Plans casual meet-ups with friends at the rental.",
        "Invites family or friends visiting the city back to the rental.",
        "Treats the rental as a social space where guests come and go.",
        "Hosts long-distance friends at the rental during the trip.",
        "Plans visitor-heavy stays; books accommodations open to it.",
        "Picks lodging that supports their habit of inviting people over.",
    ],
    "children under 10": [
        "Traveling with young children; needs family-friendly accommodation.",
        "Brings the kids along on every trip; books kid-welcoming rentals.",
        "Has young children in the travel group; values family-oriented lodging.",
        "Travels as a family unit with young ones; picks lodging that fits.",
        "Picks family resorts and kid-friendly rentals on every trip.",
        "Brings the family's young children and looks for spaces that welcome them.",
        "Plans trips around the kids' needs; needs lodging open to families.",
        "Has a young family at home and travels with them.",
        "Books lodging that fits the rhythm of traveling with small children.",
        "Travels with toddlers or grade-schoolers and chooses lodging accordingly.",
    ],
    "pets": [
        "Travels with their own dog or cat; books pet-friendly rentals.",
        "Has a beloved pet at home and brings them on trips.",
        "Considers their pet part of the family and books accommodations that welcome it.",
        "Books pet-friendly accommodations to bring their animal along.",
        "Won't leave their pet behind; picks rentals that welcome the four-legged member.",
        "Travels with the family pet; treats pet-friendly stays as a default.",
        "Picks accommodation that supports trips with their animal companion.",
        "Brings their dog or cat everywhere; wants the rental to welcome them.",
        "Animal lover who travels with their pet whenever the trip allows.",
        "Books pet-friendly lodging to keep the family pet on every trip.",
    ],
}


HOUSE_RULE_TRAITS_INVERSIONS = {
    # Inversions are mild-aversion descriptors of the LODGING preference.
    # They don't need to characterize the person personally — just
    # describe a preference for the rule being enforced.
    "smoking": [
        "Prefers smoke-free accommodations.",
        "Avoids lodging that permits smoking on the premises.",
        "Values clean-air rooms with no tobacco residue.",
        "Filters lodging for no-smoking policies.",
        "Steers away from rentals where smoking is allowed in shared spaces.",
    ],
    "parties": [
        "Prefers quiet, low-energy accommodations.",
        "Avoids party-friendly rentals when traveling.",
        "Values restful nights; filters for quiet-only properties.",
        "Steers away from lodging that hosts late-night gatherings.",
        "Prefers strict quiet-hours policies in rentals.",
    ],
    "visitors": [
        "Values privacy; prefers no-visitor lodging.",
        "Avoids accommodations open to outside visitors.",
        "Filters for strict no-visitor policies.",
        "Prefers hotel-like discretion over visitor-friendly rentals.",
        "Avoids lodging where visitors come and go freely.",
    ],
    "children under 10": [
        "Prefers adult-leaning accommodations.",
        "Avoids properties geared toward families with young children.",
        "Values a calm, mature lodging atmosphere.",
        "Filters for adult-only or age-restricted lodging.",
        "Steers away from family-heavy rentals.",
    ],
    "pets": [
        "Prefers pet-free accommodations.",
        "Avoids accommodations that allow animals on the premises.",
        "Values clean, fur-free hospitality environments.",
        "Filters for no-pet policies when booking.",
        "Steers away from lodging that welcomes other guests' pets.",
    ],
}


# --------------------------------------------------------------------------- #
# Room type  (Lifestyle + Travel Style canonical anchor)
# --------------------------------------------------------------------------- #

ROOM_TYPE_TRAITS = {
    "entire home/apt": [
        "Values space and privacy; prefers full apartment-style stays.",
        "Travels for comfort; seeks self-contained lodgings with kitchen.",
        "Prefers entire-unit rentals for group and family travel.",
        "Values the independence of an entire apartment or home.",
        "Drawn to whole-home Airbnbs with full amenities.",
        "Prefers to have an entire space rather than share common areas.",
        "Values privacy of an entire unit; doesn't want to share with strangers.",
        "Loves the home-away-from-home feel of entire rentals.",
        "Seeks accommodations large enough for multi-person or family groups.",
        "Prioritizes the convenience of an entire-place rental.",
    ],
    "private room": [
        "Comfortable with shared common areas but values a private bedroom.",
        "Prefers B&B-style private rooms with social common spaces.",
        "Balances privacy and sociability in shared lodgings.",
        "Enjoys homestays where personal space coexists with hosts.",
        "Comfortable with co-living arrangements that include a private bed.",
        "Values private bedroom while open to shared bathrooms or kitchens.",
        "Prefers the human warmth of a host alongside personal privacy.",
        "Drawn to small guesthouses with private bedrooms.",
        "Comfortable with shared kitchens but values their own room.",
        "Likes the mix of solo space and communal lounge areas.",
    ],
    "shared room": [
        "Social, budget-conscious traveler; comfortable in shared dorms.",
        "Backpacker who values cost over private space.",
        "Enjoys meeting other travelers in shared hostel rooms.",
        "Comfortable with the communal nature of shared lodging.",
        "Prioritizes adventure and connection over privacy.",
        "Drawn to youth hostels and budget-friendly shared accommodations.",
        "Values the social energy of shared bunkrooms.",
        "Comfortable with cohabitation in shared travel spaces.",
        "Travels light; happy with simple shared dorm setups.",
        "Prefers shared rooms as a way to connect with fellow travelers.",
    ],
    "not shared room": [
        "Privacy-leaning traveler; prefers solo or couple-only rooms.",
        "Avoids shared dorm-style lodgings.",
        "Values personal space; prefers non-shared accommodations.",
        "Prefers a room of their own, not one shared with strangers.",
        "Travels for rest; needs private sleeping arrangements.",
        "Avoids hostel-style shared rooms in favor of private spaces.",
        "Comfort-oriented; wants their own space at all times.",
        "Values undisturbed sleep in non-shared sleeping arrangements.",
        "Prefers private or entire-room options over shared bunks.",
        "Seeks accommodations that guarantee non-shared sleeping space.",
    ],
}


ROOM_TYPE_TRAITS_INVERSIONS = {
    "entire home/apt": [
        "Doesn't need an entire unit; comfortable with shared or smaller rentals.",
        "Finds full-apartment rentals impersonal and isolating.",
        "Prefers the human warmth of B&Bs over entire-place rentals.",
        "Considers full apartments wasteful for short stays.",
        "Doesn't enjoy the empty-apartment feel of solo entire rentals.",
    ],
    "private room": [
        "Doesn't prioritize private rooms; comfortable with shared dorms.",
        "Finds private rooms in shared properties oddly liminal.",
        "Prefers either solo apartment or fully shared dorm, not in-between.",
        "Doesn't value the B&B-style private-room compromise.",
        "Finds the privacy of a single room limiting in shared environments.",
    ],
    "shared room": [
        "Avoids shared rooms; values privacy in sleeping arrangements.",
        "Finds dorm-style accommodations uncomfortable.",
        "Prefers private space over the social energy of shared dorms.",
        "Doesn't enjoy bunk arrangements or shared sleeping spaces.",
        "Considers shared rooms invasive of personal sleep needs.",
    ],
    "not shared room": [
        "Open to shared rooms when budget allows.",
        "Comfortable with hostel-style shared sleeping arrangements.",
        "Enjoys the social dynamics of shared dorms.",
        "Doesn't need exclusive room privacy.",
        "Travels light and adapts to shared sleeping setups.",
    ],
}


# --------------------------------------------------------------------------- #
# Transportation  (Lifestyle)
# --------------------------------------------------------------------------- #

TRANSPORT_TRAITS = {
    "no flight": [
        "Eco-conscious; prefers train, bus, or road travel over flights.",
        "Carbon-aware; chooses low-emission transport options.",
        "Avoids air travel for environmental or comfort reasons.",
        "Loves scenic road trips, train journeys, and cross-country drives.",
        "Values slow travel without flights.",
        "Believes ground transportation offers a richer travel experience.",
        "Sensitive to flight emissions; prefers terrestrial alternatives.",
        "Enjoys the journey itself; flights feel rushed and impersonal.",
        "Drawn to overnight trains, scenic buses, and road-trip adventures.",
        "Prioritizes environmental sustainability in travel choices.",
    ],
    "no self-driving": [
        "Public-transit oriented; relies on transit, taxis, or rideshare.",
        "Doesn't drive; navigates cities via subway, bus, and walking.",
        "Prefers urban transit ecosystems over highway driving.",
        "Values transit-rich cities where cars aren't necessary.",
        "Enjoys the rhythm of buses, trams, and metros.",
        "Avoids car rentals; finds them stressful and expensive.",
        "Prefers walking, biking, and transit for urban exploration.",
        "Doesn't enjoy driving in unfamiliar cities; relies on transit.",
        "Drawn to walkable destinations with strong transit networks.",
        "Values flexibility of transit over the inflexibility of a rental car.",
    ],
}


TRANSPORT_TRAITS_INVERSIONS = {
    "no flight": [
        "Frequent flyer; values speed and direct flights for travel.",
        "Doesn't mind air travel; finds it the most efficient option.",
        "Prefers flights over long ground journeys.",
        "Comfortable with carbon footprint of air travel.",
        "Loves the convenience and reach of flying.",
    ],
    "no self-driving": [
        "Loves road-tripping; rents a car at every destination.",
        "Prefers the freedom of self-driving over fixed transit schedules.",
        "Confident driver; comfortable navigating unfamiliar roads.",
        "Doesn't rely on public transit; values car-based travel.",
        "Prefers the independence and flexibility of a personal vehicle.",
    ],
}


# --------------------------------------------------------------------------- #
# Attraction category  (Hobbies + Lifestyle blended) — 15 categories
# --------------------------------------------------------------------------- #

ATTRACTION_CATEGORY_TRAITS = {
    "Boat Tours & Water Sports": [
        "Water-loving; enjoys boat tours and harbor cruises.",
        "Drawn to sailing, kayaking, and open-water exploration.",
        "Loves snorkeling, diving, and underwater wildlife observation.",
        "Values waterfront destinations and aquatic recreation.",
        "Enjoys whale-watching, dolphin tours, and marine wildlife.",
        "Drawn to whitewater rafting and river adventures.",
        "Loves yacht charters, sunset cruises, and pleasure boating.",
        "Values coastal cities and harbor-front exploration.",
        "Enjoys paddleboarding, jet-skiing, and active watersports.",
        "Drawn to lakeside and oceanside destinations for water activities.",
    ],
    "Casinos & Gambling": [
        "Entertainment-district-oriented; enjoys gaming and casino floors.",
        "Drawn to Las Vegas-style energy and casino nightlife.",
        "Loves poker rooms, blackjack tables, and slot machines.",
        "Values cities with vibrant gaming and entertainment scenes.",
        "Enjoys casino-resort buffets, shows, and gaming amenities.",
        "Drawn to the spectacle of casino-floor activity.",
        "Loves the social play of card rooms and table games.",
        "Values entertainment districts with casino options.",
        "Enjoys the strategy and skill of poker tournaments.",
        "Drawn to casino-hotel accommodations and gaming nightlife.",
    ],
    "Classes & Workshops": [
        "Curious lifelong-learner; enjoys hands-on classes on trips.",
        "Drawn to language lessons, cooking workshops, and cultural classes.",
        "Loves pottery, painting, and craft-making experiences.",
        "Values learning a new skill during travel.",
        "Enjoys photography workshops and creative writing retreats.",
        "Drawn to dance classes — salsa, tango, ballet — abroad.",
        "Loves cooking-with-locals workshops and food immersion classes.",
        "Values cultural exchange through educational programming.",
        "Enjoys yoga teacher-training, meditation retreats, and wellness workshops.",
        "Drawn to artisan workshops — glassblowing, leatherwork, weaving.",
    ],
    "Concerts & Shows": [
        "Live entertainment enthusiast; enjoys theater, concerts, and comedy.",
        "Drawn to symphony, opera, and classical performance.",
        "Loves Broadway-style musicals and immersive theater.",
        "Values cities with thriving performing arts scenes.",
        "Enjoys live music festivals and outdoor concerts.",
        "Drawn to stand-up comedy clubs and intimate performance spaces.",
        "Loves dance performances — ballet, contemporary, world dance.",
        "Values the shared experience of live performance.",
        "Enjoys experimental theater and avant-garde performance.",
        "Drawn to concert halls, jazz clubs, and dance venues.",
    ],
    "Food & Drink": [
        "Foodie; enjoys culinary exploration and food-tour culture.",
        "Drawn to wine tasting, brewery tours, and craft beverages.",
        "Loves cooking classes and culinary workshops on trips.",
        "Values food markets, street-food districts, and food halls.",
        "Enjoys regional specialty tastings and farm-to-table experiences.",
        "Drawn to coffee culture and specialty bean-to-cup tasting.",
        "Loves chocolatiers, cheese shops, and artisan food makers.",
        "Values whiskey, gin, and craft-spirit distillery tours.",
        "Enjoys mixology classes and cocktail-bar exploration.",
        "Drawn to seasonal markets, wine festivals, and harvest events.",
    ],
    "Fun & Games": [
        "Playful, recreation-oriented; enjoys arcades and game rooms.",
        "Drawn to escape rooms, axe-throwing, and novelty entertainment.",
        "Loves bowling, mini-golf, and casual recreational venues.",
        "Values game-cafes and board-game gathering spots.",
        "Enjoys laser tag, paintball, and active group activities.",
        "Drawn to retro arcades and vintage gaming spaces.",
        "Loves karaoke, trivia nights, and bar-game culture.",
        "Values playful destinations with diverse recreational options.",
        "Enjoys go-karting, virtual reality, and immersive game experiences.",
        "Drawn to family-friendly entertainment centers.",
    ],
    "Museums": [
        "Curious about history and cultural heritage; enjoys docent-led tours.",
        "Drawn to photography of artifacts, paintings, and exhibition design.",
        "Finds intellectual reflection in quiet gallery spaces.",
        "Appreciates art, craftsmanship, and the stories objects carry.",
        "Values immersive learning through curated narrative.",
        "Enjoys the slow-paced contemplation of historical artifacts.",
        "Loves art history and the dialogue between eras of expression.",
        "Drawn to specialized museums — science, design, ethnography.",
        "Finds quiet inspiration in galleries and exhibition halls.",
        "Appreciates museum cafés and lingering between exhibitions.",
    ],
    "Nature & Parks": [
        "Enjoys hiking on forested trails and seeking out scenic overlooks.",
        "Drawn to wildlife observation — birds, mammals, wildflower identification.",
        "Finds renewal in wilderness immersion and quiet natural settings.",
        "Values landscape photography and the patience it requires.",
        "Loves picnicking, journaling, or simply being in green spaces.",
        "Appreciates botanical gardens and curated natural environments.",
        "Enjoys the rhythm of seasonal foliage and changing flora.",
        "Drawn to national and state parks for restorative time outdoors.",
        "Loves bird-watching, stargazing, and contemplative nature time.",
        "Finds joy in the unstructured exploration of natural spaces.",
    ],
    "Nightlife": [
        "Enjoys live music venues, jazz clubs, and rooftop bars.",
        "Drawn to vibrant downtown nightlife and social scenes.",
        "Loves dancing, late-night cocktails, and concert-going.",
        "Values cities with thriving evening cultural ecosystems.",
        "Enjoys clubbing, DJ sets, and electronic-music spaces.",
        "Drawn to speakeasy-style lounges and craft cocktail bars.",
        "Loves the energy of nightlife districts and live performance.",
        "Values the rhythm of going out late and sleeping in.",
        "Enjoys karaoke bars, pub crawls, and social-drinking traditions.",
        "Drawn to neighborhoods with after-hours music and dancing.",
    ],
    "Outdoor Activities": [
        "Active, adventure-leaning traveler; enjoys hiking and climbing.",
        "Drawn to kayaking, paddleboarding, and water-based outdoor activity.",
        "Loves cycling, biking trails, and outdoor pedaling.",
        "Values outdoor recreation as core to travel.",
        "Enjoys camping, glamping, and outdoor immersion.",
        "Drawn to rock climbing, bouldering, and adventure sports.",
        "Loves trail running, fell-walking, and mountain hiking.",
        "Values orienteering, geocaching, and exploratory outdoor games.",
        "Enjoys horseback riding, fishing, and traditional outdoor sport.",
        "Drawn to multi-sport outdoor itineraries and active vacations.",
    ],
    "Shopping": [
        "Retail experience enjoyer; loves boutiques and local artisans.",
        "Drawn to street markets and shopping districts.",
        "Enjoys fashion-week energy and designer-store browsing.",
        "Loves antique shops and vintage clothing exploration.",
        "Values boutique stores over big-box retail.",
        "Enjoys souk- or bazaar-style market browsing.",
        "Drawn to artisan crafts, ceramics, and handmade goods.",
        "Loves the discovery of one-of-a-kind shopping finds.",
        "Values local-design boutiques and pop-up retail.",
        "Drawn to outlet shopping for fashion deals.",
    ],
    "Sights & Landmarks": [
        "Architecture and history enthusiast; collects landmark photographs.",
        "Enjoys iconic monuments and the stories behind their construction.",
        "Drawn to UNESCO-listed sites and globally recognized landmarks.",
        "Loves cathedral, palace, and historic-square sightseeing.",
        "Values the symbolism and craft of landmark architecture.",
        "Enjoys walking tours that connect monuments and history.",
        "Drawn to skyline views, observation decks, and panoramic spots.",
        "Loves the sense of place that landmarks provide.",
        "Appreciates the engineering and artistry of historic structures.",
        "Photographs every landmark with care and curiosity.",
    ],
    "Spas & Wellness": [
        "Self-care-focused; values massage, sauna, and spa rituals.",
        "Drawn to wellness retreats and mineral hot springs.",
        "Enjoys yoga, meditation, and restorative spa days.",
        "Values mental and physical recovery as part of travel.",
        "Loves boutique spa experiences and aromatherapy treatments.",
        "Drawn to onsen, hammams, and global bathing traditions.",
        "Enjoys facial treatments, mud baths, and natural therapies.",
        "Values slow mornings and restorative routines on trips.",
        "Drawn to wellness-oriented hotels with full spa amenities.",
        "Finds renewal in pampering and self-care indulgence.",
    ],
    "Water & Amusement Parks": [
        "Thrill-seeking; loves water slides and lazy-river relaxation.",
        "Drawn to amusement parks for the energy and rides.",
        "Loves wave pools, splash pads, and water-park immersion.",
        "Values family-friendly amusement park itineraries.",
        "Enjoys roller coasters, drop towers, and adrenaline rides.",
        "Drawn to themed water parks with tropical immersion.",
        "Loves seasonal amusement-park visits and themed events.",
        "Values multi-day park passes and immersive experiences.",
        "Enjoys the spectacle, snacks, and crowds of amusement parks.",
        "Drawn to character meet-and-greets and themed dining at parks.",
    ],
    "Zoos & Aquariums": [
        "Animal-lover; enjoys zoo visits and wildlife observation.",
        "Drawn to aquariums and marine-life education.",
        "Loves zoological conservation and rehabilitation programs.",
        "Values wildlife encounters in safe, educational settings.",
        "Enjoys safari parks, big-cat reserves, and tropical aviaries.",
        "Drawn to penguin habitats, dolphin tanks, and shark exhibits.",
        "Loves children's zoos and family animal experiences.",
        "Values the storytelling of zoo and aquarium docents.",
        "Enjoys photographing exotic animals up close.",
        "Drawn to wildlife conservancies and breeding programs.",
    ],
}


ATTRACTION_CATEGORY_TRAITS_INVERSIONS = {
    "Boat Tours & Water Sports": [
        "Avoids water activities; doesn't enjoy boat-based travel.",
        "Doesn't swim or engage with water sports.",
        "Finds boat tours touristy or overly long.",
        "Prefers landlocked destinations over coastal travel.",
        "Doesn't value waterfront recreation.",
    ],
    "Casinos & Gambling": [
        "Avoids casinos; finds gambling unappealing.",
        "Doesn't enjoy casino-resort culture; prefers quieter places.",
        "Finds gaming floors loud and over-stimulating.",
        "Prefers cultural travel to casino entertainment.",
        "Avoids gambling and games-of-chance environments.",
    ],
    "Classes & Workshops": [
        "Avoids structured classes on travel; prefers free-flow time.",
        "Doesn't enjoy workshop formats; finds them limiting.",
        "Prefers unstructured exploration to scheduled learning.",
        "Finds classes interruptive to relaxation.",
        "Doesn't engage with workshop-based travel.",
    ],
    "Concerts & Shows": [
        "Doesn't enjoy live performance; finds shows hard to focus on.",
        "Avoids concerts and theater; prefers quiet entertainment.",
        "Prefers recorded media over live performance.",
        "Finds concert venues loud and crowded.",
        "Doesn't enjoy the ritual of attending live shows.",
    ],
    "Food & Drink": [
        "Doesn't enjoy food-focused travel; eats simply on trips.",
        "Avoids elaborate food tours; prefers casual eats.",
        "Finds wine and brewery culture pretentious.",
        "Prefers attractions over food-centric experiences.",
        "Doesn't value culinary exploration as a travel goal.",
    ],
    "Fun & Games": [
        "Doesn't enjoy arcades or game centers on trips.",
        "Avoids game-cafes and recreational entertainment.",
        "Prefers cultural depth over playful entertainment.",
        "Finds bowling and similar leisure tedious.",
        "Doesn't engage with novelty entertainment venues.",
    ],
    "Museums": [
        "Finds museums dry and over-curated; prefers outdoor experiences.",
        "Doesn't enjoy quiet gallery contemplation; needs more active engagement.",
        "Avoids museums; finds them stuffy and over-scheduled.",
        "Prefers experiential learning over museum-style display.",
        "Finds historical artifacts uninteresting; gravitates elsewhere.",
    ],
    "Nature & Parks": [
        "Prefers urban environments to nature parks.",
        "Doesn't enjoy outdoor hiking; finds nature trails tiring.",
        "Avoids parks; prefers indoor cultural experiences.",
        "Finds wilderness immersion uncomfortable; needs amenities nearby.",
        "Doesn't engage with natural settings; prefers built environments.",
    ],
    "Nightlife": [
        "Avoids nightlife; prefers quiet, restorative evenings.",
        "Doesn't enjoy bars, clubs, or late-night socializing.",
        "Prefers early bedtimes over late-night party districts.",
        "Finds nightlife loud and overstimulating.",
        "Values calm evenings over crowded entertainment venues.",
    ],
    "Outdoor Activities": [
        "Avoids outdoor recreation; prefers indoor cultural travel.",
        "Doesn't enjoy hiking or active outdoor sports.",
        "Finds adventure sports physically uncomfortable.",
        "Prefers museum-based travel over outdoor activity.",
        "Avoids strenuous outdoor itineraries.",
    ],
    "Shopping": [
        "Avoids shopping districts; prefers experiences to purchases.",
        "Doesn't enjoy retail or boutique browsing.",
        "Finds shopping-as-travel materialistic.",
        "Prefers cultural sites over shopping streets.",
        "Doesn't shop on trips; travels for experiences only.",
    ],
    "Sights & Landmarks": [
        "Finds landmark-checking tourist-trap-y; prefers off-the-beaten-path.",
        "Doesn't enjoy iconic photo-op sites; prefers authentic local spaces.",
        "Avoids crowded landmarks; finds them overhyped.",
        "Prefers neighborhoods and everyday life over monumental architecture.",
        "Finds famous landmarks impersonal and predictable.",
    ],
    "Spas & Wellness": [
        "Doesn't enjoy spas; finds them too quiet or self-indulgent.",
        "Prefers active travel over restorative spa days.",
        "Avoids massage and pampering; not their style of relaxation.",
        "Finds wellness retreats overly precious.",
        "Prefers gritty exploration to wellness amenities.",
    ],
    "Water & Amusement Parks": [
        "Avoids amusement parks; finds them overstimulating.",
        "Doesn't enjoy water slides or wave pools.",
        "Finds theme parks crowded and overpriced.",
        "Prefers cultural travel to amusement-park itineraries.",
        "Avoids family-resort-style park settings.",
    ],
    "Zoos & Aquariums": [
        "Doesn't enjoy zoos; finds them sad or commercial.",
        "Avoids aquariums; prefers wild observation over confined animals.",
        "Doesn't engage with zoo-based travel.",
        "Finds zoo visits tedious for adults.",
        "Prefers wildlife encounters in natural habitat.",
    ],
}


# --------------------------------------------------------------------------- #
# Numeric optimization-direction traits  (Lifestyle / Travel Style cues)
# Keyed by (entity, attribute, direction) where direction ∈ {"max", "min"}.
# `max` means the user wants the value as high as possible (op >=);
# `min` means the user wants it as low as possible (op <=).
# Aggregation (avg/min/max/sum) for NumericPreference can be appended by
# the generator as a modifier — the underlying trait is the same.
# --------------------------------------------------------------------------- #

NUMERIC_DIRECTION_TRAITS = {
    ("Accommodation", "rating", "max"): [
        "Comfort-prioritizing traveler; values high-rated accommodations.",
        "Drawn to boutique hotels with strong guest reviews.",
        "Values polished service, quality bedding, and well-rated stays.",
        "Reads accommodation reviews carefully before booking.",
        "Discerning about lodging quality; pursues top-rated properties.",
        "Loves the assurance of a well-rated hotel or rental.",
        "Believes a good night's sleep starts with a well-rated room.",
        "Drawn to luxury or near-luxury accommodation tiers.",
        "Values the consistency of well-reviewed properties.",
        "Prefers four- or five-star properties for restful nights.",
    ],
    ("Accommodation", "rating", "min"): [
        "Casual about accommodation quality; comfortable in basic lodgings.",
        "Doesn't prioritize hotel ratings; values cost and convenience.",
        "Comfortable in budget motels and economy properties.",
        "Low-maintenance traveler; sleeps anywhere clean and safe.",
        "Values cost savings over polished accommodation experience.",
        "Comfortable in hostels, budget B&Bs, and simple lodgings.",
        "Doesn't sweat hotel-rating details; flexible about quality.",
        "Drawn to affordable, no-frills accommodation options.",
        "Believes lodging is just a place to sleep; ratings matter little.",
        "Prefers minimalist, unfussy accommodation experiences.",
    ],
    ("Accommodation", "cost", "max"): [
        "Comfortable with premium accommodation pricing.",
        "Drawn to luxury hotels and resort-style lodging.",
        "Values polished, full-service properties at premium tiers.",
        "Prefers boutique hotels and high-end vacation rentals.",
        "Comfortable spending generously on premium stays.",
        "Drawn to luxury resorts and signature hospitality brands.",
        "Values the indulgence of high-end accommodation.",
        "Prefers premium lodging with full-service amenities.",
        "Comfortable with high-end hotel pricing for quality.",
        "Drawn to flagship luxury properties and resort destinations.",
    ],
    ("Accommodation", "cost", "min"): [
        "Budget-conscious traveler; minimizes accommodation spend.",
        "Drawn to affordable hostels, motels, and inexpensive lodgings.",
        "Values cost savings on rooms to spend more on experiences.",
        "Comfortable in low-cost accommodations to extend travel budget.",
        "Prefers cheap stays even if amenities are minimal.",
        "Drawn to budget-tier hotels and discount lodging deals.",
        "Believes accommodation is a place to sleep, not splurge on.",
        "Values economical lodging for longer or more frequent trips.",
        "Comfortable in basic budget properties.",
        "Drawn to value-oriented lodging choices.",
    ],

    ("Restaurant", "rating", "max"): [
        "Discerning diner who values culinary quality and reviews.",
        "Drawn to top-rated restaurants and food-critic favorites.",
        "Values fine dining and chef-driven menus.",
        "Prefers Michelin-recognized or highly-reviewed dining.",
        "Loves the artistry of well-rated kitchens.",
        "Drawn to multi-course tasting menus at acclaimed restaurants.",
        "Values culinary excellence and high-rated dining experiences.",
        "Prefers top-tier dining; reads restaurant reviews carefully.",
        "Comfortable spending more for highly-reviewed cuisine.",
        "Discerning palate; seeks out the most acclaimed restaurants.",
    ],
    ("Restaurant", "rating", "min"): [
        "Casual diner; comfortable with basic eateries.",
        "Doesn't prioritize restaurant ratings; values cost and convenience.",
        "Drawn to street food, hole-in-the-wall spots, and inexpensive meals.",
        "Comfortable in budget restaurants and fast-casual venues.",
        "Low-maintenance about dining; eats wherever's convenient.",
        "Values food as fuel; ratings don't drive choices.",
        "Comfortable in modest restaurants and casual cafés.",
        "Doesn't sweat restaurant-rating details; flexible about dining.",
        "Drawn to affordable food trucks and cheap-eats districts.",
        "Believes meals are nourishment; ratings are secondary.",
    ],
    ("Restaurant", "cost", "max"): [
        "Comfortable with premium dining pricing.",
        "Drawn to fine dining and tasting-menu restaurants.",
        "Values polished, multi-course dining at premium tiers.",
        "Prefers Michelin-style restaurants and signature kitchens.",
        "Comfortable spending generously on memorable meals.",
        "Drawn to chef's tables and high-end culinary destinations.",
        "Values the indulgence of premium dining experiences.",
        "Prefers premium restaurants with full-service amenities.",
        "Comfortable with high-end menu pricing for quality.",
        "Drawn to flagship restaurants and celebrity-chef venues.",
    ],
    ("Restaurant", "cost", "min"): [
        "Budget-conscious eater; minimizes restaurant spending.",
        "Drawn to cheap eats, street food, and inexpensive dining.",
        "Values cost-effective meals to extend travel budget.",
        "Comfortable in low-cost restaurants and food markets.",
        "Prefers cheap-eats traditions over fine dining.",
        "Drawn to value menus, daily specials, and discounts.",
        "Believes food should be cheap and abundant.",
        "Values economical meals for longer trips.",
        "Comfortable in modest, budget-friendly eateries.",
        "Drawn to inexpensive food districts and casual takeout.",
    ],

    ("Attraction", "rating", "max"): [
        "Discerning sightseer; visits only top-rated attractions.",
        "Drawn to acclaimed museums, landmarks, and venues.",
        "Values the curated quality of high-rated attractions.",
        "Prefers attractions with strong reviews and visitor feedback.",
        "Reads attraction reviews carefully before visiting.",
        "Loves the assurance of widely-acclaimed attractions.",
        "Values attraction quality over quantity in itineraries.",
        "Drawn to top-tier museums, parks, and landmarks.",
        "Discerning visitor; seeks out the most acclaimed sites.",
        "Believes great attractions reward selective travel itineraries.",
    ],
    ("Attraction", "rating", "min"): [
        "Casual sightseer; comfortable with basic attractions.",
        "Doesn't prioritize attraction ratings; values exploration.",
        "Drawn to off-the-beaten-path, lesser-known venues.",
        "Comfortable in modest attractions and small museums.",
        "Low-maintenance traveler; visits whatever's nearby.",
        "Values discovery over acclaim in attractions.",
        "Comfortable in small, unrated local venues.",
        "Doesn't sweat attraction-rating details.",
        "Drawn to authentic, less-touristed spots.",
        "Believes attractions need not be famous to be meaningful.",
    ],

    ("Transportation", "cost", "max"): [
        "Comfortable with premium transportation pricing.",
        "Drawn to business-class flights and first-class rail.",
        "Values polished, fast, premium transit.",
        "Prefers private transfers and chauffeured rides.",
        "Comfortable spending generously on travel comfort.",
        "Drawn to premium-class transit and signature transport.",
        "Values the indulgence of premium transit experiences.",
        "Prefers premium transportation with full-service amenities.",
        "Comfortable with high-end transit pricing for quality.",
        "Drawn to first-class transport and signature airline lounges.",
    ],
    ("Transportation", "cost", "min"): [
        "Budget-conscious about transportation; minimizes transit spend.",
        "Drawn to bus, train, or low-cost flight options.",
        "Values cost-effective movement between destinations.",
        "Comfortable with economy transit and budget travel options.",
        "Prefers cheap transit over premium-class travel.",
        "Drawn to value-tier transportation choices.",
        "Believes transit is just connection; minimizes its cost.",
        "Values economical transit for longer-haul trips.",
        "Comfortable with basic, no-frills travel modes.",
        "Drawn to budget transit options like buses and economy flights.",
    ],
}


NUMERIC_DIRECTION_TRAITS_INVERSIONS = {
    ("Accommodation", "rating", "max"): [
        "Doesn't prioritize hotel ratings; comfortable in basic lodgings.",
        "Finds luxury accommodations impersonal and over-the-top.",
        "Avoids high-rated properties; finds them stiff.",
        "Believes hotel ratings are overrated; cares about cost more.",
        "Doesn't enjoy the formality of upscale lodging.",
    ],
    ("Accommodation", "rating", "min"): [
        "Avoids budget lodgings; values comfort and reviews.",
        "Doesn't enjoy basic motels; finds them uncomfortable.",
        "Prefers higher-rated lodging even at extra cost.",
        "Finds low-rated properties anxiety-inducing.",
        "Won't compromise on accommodation quality.",
    ],
    ("Accommodation", "cost", "max"): [
        "Avoids premium lodging; values cost savings.",
        "Doesn't enjoy luxury hotels; finds them ostentatious.",
        "Prefers economical accommodation choices.",
        "Finds high-end lodging wasteful for short stays.",
        "Won't pay premium prices for accommodation.",
    ],
    ("Accommodation", "cost", "min"): [
        "Doesn't prioritize budget lodging; values quality.",
        "Avoids hostels and economy properties.",
        "Believes accommodation is worth investing in.",
        "Finds cheap lodging unpleasant and noisy.",
        "Prefers full-service hotels even at premium cost.",
    ],
    ("Restaurant", "rating", "max"): [
        "Doesn't prioritize restaurant ratings; comfortable in basic eateries.",
        "Finds fine dining pretentious and over-the-top.",
        "Avoids Michelin-ranked spots; prefers casual places.",
        "Believes restaurant ratings are inflated; ignores them.",
        "Doesn't enjoy the formality of high-end dining.",
    ],
    ("Restaurant", "rating", "min"): [
        "Avoids low-rated restaurants; values reviews carefully.",
        "Doesn't enjoy hole-in-the-wall spots; finds them risky.",
        "Prefers higher-rated dining even at extra cost.",
        "Finds low-rated eateries unpleasant.",
        "Won't compromise on dining quality.",
    ],
    ("Restaurant", "cost", "max"): [
        "Avoids fine dining; values cost savings.",
        "Doesn't enjoy expensive restaurants; finds them stuffy.",
        "Prefers economical dining choices.",
        "Finds high-end meals wasteful for daily travel.",
        "Won't pay premium prices for restaurants.",
    ],
    ("Restaurant", "cost", "min"): [
        "Doesn't prioritize cheap eats; values dining quality.",
        "Avoids street food and budget dining.",
        "Believes meals are worth investing in.",
        "Finds cheap restaurants unpleasant.",
        "Prefers fine dining even at premium cost.",
    ],
    ("Attraction", "rating", "max"): [
        "Doesn't prioritize attraction ratings; comfortable with any sites.",
        "Finds famous attractions impersonal and over-the-top.",
        "Avoids top-rated landmarks; prefers off-beaten paths.",
        "Believes attraction ratings are inflated.",
        "Doesn't enjoy the crowds at acclaimed sites.",
    ],
    ("Attraction", "rating", "min"): [
        "Avoids low-rated attractions; values reviews.",
        "Doesn't enjoy small, unrated venues.",
        "Prefers higher-rated attractions even with crowds.",
        "Finds low-rated sites underwhelming.",
        "Won't compromise on attraction quality.",
    ],
    ("Transportation", "cost", "max"): [
        "Avoids premium transit; values cost savings.",
        "Doesn't enjoy first-class travel; finds it wasteful.",
        "Prefers economical transit choices.",
        "Finds business-class wasteful for short trips.",
        "Won't pay premium prices for transit.",
    ],
    ("Transportation", "cost", "min"): [
        "Doesn't prioritize cheap transit; values comfort.",
        "Avoids budget transit and economy travel.",
        "Believes transit is worth investing in.",
        "Finds cheap transit cramped and uncomfortable.",
        "Prefers premium-class travel even at higher cost.",
    ],
}


# --------------------------------------------------------------------------- #
# Atomic numeric predicate traits  (quartile-anchored).
# Keyed by (entity, attribute, op, quartile) where:
#   * op       ∈ {">=", "<="}
#   * quartile ∈ {"Q1", "Q2", "Q3", None}
#     Q1 = cheap / low-end of the data distribution
#     Q2 = median / balanced
#     Q3 = costly / high-end
#     None = bank entry has no stat-based anchor (direction only)
# Semantic reading:
#   * "cost <= Qk" — caps spend at the Qk tier   (lower k = more frugal,
#                                                  higher k = price-tolerant)
#   * "cost >= Qk" — floors spend at the Qk tier (high k = splurger)
#   * "rating >= Qk" — floors quality at the Qk tier (high k = demanding)
#   * "rating <= Qk" — caps rating at the Qk tier   (anti-polish / off-radar)
# Keys are populated only for combinations actually present in the
# augmented corpus.
# --------------------------------------------------------------------------- #

ATOMIC_NUMERIC_TRAITS = {
    ("Accommodation", "cost", "<=", "Q1"): [
        "Books lodging at the cheap end of the local market.",
        "Caps room spend at budget-tier rates everywhere they go.",
        "Treats accommodation as a place to sleep, not splurge on.",
        "Hunts down the lowest viable nightly rate in each city.",
        "Sticks to hostel- and motel-tier lodging to free cash for the trip.",
        "Avoids paying more than the bottom of the price range for a room.",
        "Optimizes the lodging line item hard; aims for rock-bottom prices.",
        "Considers hotels above budget-tier a poor use of trip money.",
        "Plans around the cheapest available rooms first, then everything else.",
        "Prefers basic, low-cost rooms over amenities or polish.",
    ],
    ("Accommodation", "cost", "<=", "Q2"): [
        "Keeps lodging spend in the middle of the local market.",
        "Targets mid-range hotels — not bargain, not boutique.",
        "Balances comfort and cost on the room line; mid-tier suits them.",
        "Books moderately-priced rooms without going premium.",
        "Avoids both bottom-tier dives and luxury overspend on lodging.",
        "Caps lodging cost at the median rate for the city.",
        "Pays standard mid-market hotel rates; no need for luxury.",
        "Sits in the middle of the price range for accommodation.",
        "Picks mid-priced rooms as the comfortable balance.",
        "Considers median accommodation rates the natural spend.",
    ],
    ("Accommodation", "cost", "<=", "Q3"): [
        "Doesn't sweat lodging costs — anything up to upscale is fine.",
        "Loose price ceiling on rooms; only blocks truly extravagant spend.",
        "Comfortable paying through to the upper-tier rate on accommodation.",
        "Books across the price range without picking the absolute cheapest.",
        "Cost-tolerant on lodging — willing to go up to high-end but not exceed it.",
        "Caps the room budget at the high end of the market, no further.",
        "Accommodation spend is flexible up to the upscale-but-not-luxury tier.",
        "Open to most lodging tiers as long as it stops short of luxury extremes.",
        "Will spend up to but not beyond the upper bracket on rooms.",
        "Price-flexible across nearly all lodging tiers.",
    ],
    ("Accommodation", "cost", ">=", "Q3"): [
        "Strongly prefers premium-priced lodging — luxury or near-luxury.",
        "Splurges on top-bracket hotels and resort properties.",
        "Floor on room rates sits at the upscale end of the market.",
        "Books primarily in the high-cost tier for accommodations.",
        "Treats premium spend on rooms as part of the experience.",
        "Aims to stay at the upscale tier for lodging in most cities.",
        "Considers high accommodation spend a signal of trip quality.",
        "Picks expensive rooms by design, not by accident.",
        "Sets a premium spending floor on every room booked.",
        "Drawn to the top of the local accommodation price range.",
    ],

    ("Accommodation", "rating", ">=", "Q2"): [
        "Wants above-median guest ratings on every booked stay.",
        "Floors lodging quality at the median review tier.",
        "Avoids weakly-reviewed properties; looks for solid scores.",
        "Steers toward hotels with at-least-decent guest feedback.",
        "Won't book a room rated below the local median.",
        "Filters lodging on review scores at the median or better.",
        "Looks for a respectable rating bar on accommodation.",
        "Books only rooms whose guest scores clear the median bar.",
        "Considers median-and-up review scores the minimum on rooms.",
        "Treats above-average ratings as the lodging baseline.",
    ],
    ("Accommodation", "rating", ">=", "Q3"): [
        "Strongly prefers top-tier guest ratings on every lodging choice.",
        "Leans toward highly-rated, well-reviewed properties.",
        "Sets a rating floor at the upper review-score tier for rooms.",
        "Picks accommodations primarily from the top of the review distribution.",
        "Aims to book stays in the upper-tier rating bracket.",
        "Treats top-rated hotels as the natural lodging baseline.",
        "Seeks out the highest-scoring accommodation in each city.",
        "Filters lodging on review scores; favors the top performers.",
        "Considers strong upper-tier ratings the comfortable threshold.",
        "Picks rooms from the most-acclaimed end of the market.",
    ],
    ("Accommodation", "rating", "<=", "Q2"): [
        "Caps accommodation rating below the top tier — anti-polish.",
        "Avoids review-darling hotels; picks lower-rated, off-radar stays.",
        "Steers away from polished, highly-rated properties on principle.",
        "Books modestly-rated lodging — less-vetted, more local-feeling.",
        "Looks past the top of the review distribution for accommodation.",
        "Prefers under-the-radar stays that don't top the ratings list.",
        "Comfortable in lodging the review crowd hasn't blessed.",
        "Avoids the polish of high-rated hotels; books smaller, modest places.",
        "Books accommodation from the lower-rated end of the market.",
        "Treats moderate-or-lower ratings as a feature of authentic lodging.",
    ],
    ("Accommodation", "rating", ">=", None): [
        "Leans toward higher-rated lodging in general; review scores matter.",
        "Wants reasonable review scores on accommodation, no fixed bar.",
        "Pulls toward better-rated stays when comparing options.",
        "Cares about lodging ratings even without a strict threshold.",
        "Reads guest reviews and tilts toward stronger feedback.",
        "Picks the better-rated room when picking between comparable stays.",
        "Considers ratings a positive signal when booking accommodation.",
        "Defaults to higher-scoring hotels among similarly-priced options.",
        "Treats stronger reviews as a tiebreaker on lodging picks.",
        "Slants accommodation choices toward better-reviewed properties.",
    ],
    ("Accommodation", "rating", "<=", None): [
        "Doesn't chase lodging ratings; modest reviews are fine.",
        "Indifferent to accommodation review scores in general.",
        "Picks lodging without weighting review tier heavily.",
        "Comfortable in low- or middle-rated accommodation overall.",
        "Doesn't filter rooms by ratings when planning a trip.",
        "Stays at the cheaper option even if its rating is lower.",
        "Considers room ratings secondary to cost and location.",
        "Books accommodation regardless of where it falls on the rating curve.",
        "Doesn't weight guest scores when choosing rooms.",
        "Treats ratings as nice-to-know, not load-bearing on lodging.",
    ],

    ("Restaurant", "cost", "<=", "Q1"): [
        "Sticks to inexpensive eateries — street food and budget meals.",
        "Caps restaurant spend at the bottom of the local market.",
        "Hunts cheap-eats districts and value-tier menus.",
        "Eats well below the local median check size.",
        "Avoids spending more than budget-tier prices on a meal.",
        "Treats food as fuel; keeps every restaurant bill cheap.",
        "Leans toward the cheapest dining options each city offers.",
        "Optimizes meal spend hard; expects rock-bottom checks.",
        "Defaults to food trucks, holes-in-the-wall, and budget cafés.",
        "Prefers cheap meals over mid- or upscale dining.",
    ],
    ("Restaurant", "cost", "<=", "Q2"): [
        "Keeps meal spend at the city's median check size.",
        "Targets mid-priced restaurants — not cheap eats, not fine dining.",
        "Balances dining cost and quality at the middle of the market.",
        "Books neighborhood bistros and standard mid-tier restaurants.",
        "Avoids both bargain spots and white-tablecloth pricing on meals.",
        "Caps dining cost at the median for the destination.",
        "Pays standard mid-market check averages; no extremes.",
        "Sits in the middle of the price range when dining out.",
        "Picks moderately-priced restaurants as a deliberate balance.",
        "Considers median menu prices the natural meal spend.",
    ],
    ("Restaurant", "cost", "<=", "Q3"): [
        "Loose price ceiling on meals; only blocks the most extravagant.",
        "Comfortable paying through to the upper-tier check size on meals.",
        "Cost-tolerant on dining — willing to go upscale but not extremes.",
        "Caps the meal budget at the high end of the market, no further.",
        "Doesn't sweat menu prices up to the upscale tier.",
        "Books restaurants across the price spectrum except the very top.",
        "Open to most restaurant tiers short of the truly extravagant.",
        "Will pay up to the high-end bracket on meals but not beyond.",
        "Price-flexible across nearly all dining tiers.",
        "Treats upper-tier check sizes as the meal-cost cap.",
    ],
    ("Restaurant", "cost", ">=", "Q2"): [
        "Floors dining spend at the median check size of the local market.",
        "Picks mid-tier-or-above restaurants; skips the cheap-eats stratum.",
        "Sets a floor on meal spend at the city's median price.",
        "Considers below-median-priced restaurants too budget for the trip.",
        "Prefers proper sit-down meals over bargain-tier dining.",
        "Aims to book restaurants at or above the median price tier.",
        "Treats median check sizes as the comfortable baseline for a real meal.",
        "Spends at least median-tier on meals; values the step up from cheap.",
        "Picks mid-priced restaurants as the deliberate floor on dining.",
        "Skips bargain spots; floors meal spend at the median.",
    ],
    ("Restaurant", "cost", ">=", "Q3"): [
        "Splurges on meals; books at the top of the local price range.",
        "Floors dining spend at the upscale check tier.",
        "Strongly prefers premium-priced restaurants — fine dining.",
        "Treats every meal as an occasion worth the premium spend.",
        "Aims to book restaurants at the upper-tier check size.",
        "Picks expensive meals by design — high-end menus.",
        "Sets a premium spending floor on every restaurant booking.",
        "Drawn to chef-driven, top-of-market dining destinations.",
        "Considers high meal spend part of the trip's signature.",
        "Books primarily from the priciest tier of restaurants.",
    ],

    ("Restaurant", "rating", ">=", "Q2"): [
        "Wants above-median ratings on every restaurant.",
        "Floors dining quality at the median review tier.",
        "Avoids weakly-reviewed eateries; looks for solid scores.",
        "Steers toward restaurants with at-least-decent feedback.",
        "Aims to dine at restaurants rated at or above the local median.",
        "Filters dining on review scores at the median or better.",
        "Looks for a respectable rating bar on every restaurant booked.",
        "Picks restaurants whose review scores clear the median bar.",
        "Considers median-and-up review scores the comfortable baseline.",
        "Treats above-average ratings as the dining baseline.",
    ],
    ("Restaurant", "rating", ">=", "Q3"): [
        "Strongly prefers top-tier ratings on every restaurant.",
        "Leans toward highly-rated, food-critic-favored dining.",
        "Sets a rating floor at the upper review-score tier for meals.",
        "Picks restaurants primarily from the top of the review distribution.",
        "Aims to dine in the upper-tier rating bracket.",
        "Treats top-rated restaurants as the natural dining baseline.",
        "Seeks out the highest-scoring restaurants in every city.",
        "Filters dining on reviews; favors the top performers.",
        "Considers strong upper-tier ratings the comfortable threshold for meals.",
        "Picks meals at the most-acclaimed end of the local market.",
    ],
    ("Restaurant", "rating", "<=", "Q3"): [
        "Doesn't insist on top-rated meals — anything up to upper-tier is fine.",
        "Caps restaurant rating below the absolute peak; comfortable with most spots.",
        "Open to dining across the rating spectrum short of the very top.",
        "Doesn't filter restaurants on top-tier reviews; flexible about quality.",
        "Books meals across the rating range without chasing acclaim.",
        "Treats top-tier acclaim as nice-but-unnecessary in restaurants.",
        "Comfortable eating at moderately-rated places, not just the best.",
        "Doesn't restrict dining to the most-reviewed restaurants in town.",
        "Picks restaurants without demanding the highest local ratings.",
        "Open about dining quality; not gunning for the rating peak.",
    ],

    ("Attraction", "rating", ">=", "Q2"): [
        "Wants well-reviewed attractions; above-median scores at minimum.",
        "Floors attraction quality at the median visitor-feedback tier.",
        "Avoids poorly-reviewed venues; looks for solid scores.",
        "Steers toward attractions with at-least-decent ratings.",
        "Won't visit an attraction rated below the local median.",
        "Filters sights on review scores at the median or better.",
        "Picks attractions whose visitor scores clear the median bar.",
        "Considers median-and-up ratings the minimum on sights.",
        "Treats above-average ratings as the attraction baseline.",
        "Skips weakly-reviewed venues but doesn't chase the absolute top.",
    ],
    ("Attraction", "rating", ">=", "Q3"): [
        "Strongly prefers top-rated attractions; leans toward the most acclaimed.",
        "Books sightseeing primarily at the upper end of the review distribution.",
        "Sets a rating floor at the top tier for every attraction visited.",
        "Aims to spend itinerary time on attractions at upper-tier ratings.",
        "Treats top-rated landmarks and museums as the standout itinerary stops.",
        "Seeks out the highest-scoring attractions in each city.",
        "Filters sights on review scores; favors the top performers.",
        "Considers upper-tier ratings the comfortable bar for an attraction visit.",
        "Picks attractions from the most-acclaimed end of the local list.",
        "Plans itineraries that lean heavily on top-rated sights.",
    ],

    ("Transportation", "cost", "<=", "Q1"): [
        "Books the cheapest available transit between destinations.",
        "Caps transport spend at the bottom of the market.",
        "Goes for budget airlines, economy rail, and bus routes.",
        "Avoids spending more than budget-tier rates to get around.",
        "Treats transit as connection only; minimizes every dollar.",
        "Hunts cheap fares between cities, even at the cost of comfort.",
        "Defaults to the least-expensive way of moving between stops.",
        "Optimizes transport spend hard; aims for rock-bottom fares.",
        "Books bus, low-cost flights, and budget rail by default.",
        "Prefers cheap travel modes over comfort or speed.",
    ],
    ("Transportation", "cost", ">=", "Q3"): [
        "Strongly prefers premium-tier transit — business class or private transfers.",
        "Floors transport spend at the upscale travel tier.",
        "Aims to book transit at the high-end of the local market.",
        "Splurges on premium transportation for every leg of the trip.",
        "Treats high transit spend as part of the travel experience.",
        "Picks first-class flights, premium rail, and chauffeured transfers.",
        "Sets a premium spending floor on transport every time.",
        "Drawn to top-bracket airline classes and signature transport.",
        "Considers expensive transit a signature of trip comfort.",
        "Books primarily from the priciest tier of transport options.",
    ],
}


ATOMIC_NUMERIC_TRAITS_INVERSIONS = {
    ("Accommodation", "cost", "<=", "Q1"): [
        "Won't compromise on hotels; comfortable spending above budget-tier.",
        "Avoids cheap rooms; finds them noisy and unpleasant.",
        "Prefers paying for amenities rather than the bottom rate.",
        "Believes accommodation is worth investing in.",
        "Books mid- or upper-tier hotels even on tight budgets.",
    ],
    ("Accommodation", "cost", "<=", "Q2"): [
        "Skips mid-tier rooms; books either rock-bottom or upscale.",
        "Finds the middle of the lodging market dull — needs an extreme.",
        "Won't settle for moderately-priced hotels; goes high or low.",
        "Bimodal on rooms — either budget hostels or luxury suites.",
        "Avoids median-priced accommodations on principle.",
    ],
    ("Accommodation", "cost", "<=", "Q3"): [
        "Keeps lodging spend tight; wouldn't touch upper-tier rates.",
        "Sensitive about hotel costs — even mid-upper bracket feels wasteful.",
        "Caps room spend well below the upscale tier.",
        "Refuses to pay close to top-bracket rates for accommodation.",
        "Avoids any room that strays toward the upper end of the market.",
    ],
    ("Accommodation", "cost", ">=", "Q3"): [
        "Never books premium hotels; finds luxury wasteful.",
        "Avoids high-end lodging entirely; sticks to budget tiers.",
        "Won't pay luxury rates for somewhere to sleep.",
        "Considers upscale accommodation indulgent and pointless.",
        "Books cheap rooms even when budget allows premium.",
    ],
    ("Accommodation", "rating", ">=", "Q2"): [
        "Doesn't weight lodging ratings; books regardless of reviews.",
        "Comfortable in modestly-rated rooms below the median tier.",
        "Skips the review-filter step when picking accommodation.",
        "Stays at low-rated lodging without concern.",
        "Doesn't insist on median-or-better ratings on rooms.",
    ],
    ("Accommodation", "rating", ">=", "Q3"): [
        "Doesn't chase top-rated accommodation; mid-range is fine.",
        "Avoids the polish of acclaimed hotels; prefers modest stays.",
        "Skips the top-of-rating-curve step on lodging.",
        "Considers top-tier review scores overkill on rooms.",
        "Comfortable in moderately-rated accommodation.",
    ],
    ("Accommodation", "rating", "<=", "Q2"): [
        "Insists on well-reviewed lodging; won't touch low-rated rooms.",
        "Avoids modest-rated stays; books only above-median properties.",
        "Considers low ratings a non-starter for accommodation.",
        "Won't risk under-the-radar lodging without strong reviews.",
        "Demands review-vetted hotels rather than off-radar stays.",
    ],
    ("Accommodation", "rating", ">=", None): [
        "Doesn't weight lodging ratings at all; cost beats reviews.",
        "Picks rooms by price or location, ignoring review tier.",
        "Considers ratings a sideshow when booking accommodation.",
        "Avoids the review-comparison step entirely on rooms.",
        "Doesn't tilt toward better-reviewed stays at all.",
    ],
    ("Accommodation", "rating", "<=", None): [
        "Prefers higher-rated accommodation even without a strict bar.",
        "Tilts toward better-reviewed hotels in general.",
        "Cares about lodging ratings as a positive signal.",
        "Picks the higher-rated room among comparable options.",
        "Considers stronger reviews a meaningful nudge on rooms.",
    ],

    ("Restaurant", "cost", "<=", "Q1"): [
        "Avoids cheap eats; finds budget meals unsatisfying.",
        "Won't settle for street food or budget restaurants.",
        "Prefers paying for a proper meal over a quick cheap bite.",
        "Believes good food is worth a real check size.",
        "Books mid- or upper-tier restaurants even on tight budgets.",
    ],
    ("Restaurant", "cost", "<=", "Q2"): [
        "Skips mid-tier restaurants; books either cheap eats or fine dining.",
        "Finds the middle of the restaurant market dull — wants an extreme.",
        "Won't settle for moderately-priced meals; goes high or low.",
        "Bimodal on dining — either food trucks or signature kitchens.",
        "Avoids median-priced restaurants on principle.",
    ],
    ("Restaurant", "cost", "<=", "Q3"): [
        "Keeps meal spend tight; wouldn't touch upper-tier check sizes.",
        "Sensitive about restaurant prices — even mid-upper feels wasteful.",
        "Caps dining spend well below the upscale tier.",
        "Refuses to pay close to top-bracket prices for a meal.",
        "Avoids any restaurant near the upper end of the market.",
    ],
    ("Restaurant", "cost", ">=", "Q2"): [
        "Doesn't insist on median-priced restaurants; cheap eats are fine.",
        "Books across the full price spectrum, including bargain spots.",
        "Comfortable dining at the bottom of the restaurant market.",
        "Doesn't set a floor on meal spend; budget eateries work.",
        "Considers cheap-eats spots part of the travel experience.",
    ],
    ("Restaurant", "cost", ">=", "Q3"): [
        "Never splurges on meals; finds fine dining wasteful.",
        "Avoids premium restaurants entirely; sticks to budget eateries.",
        "Won't pay top-bracket prices for a single meal.",
        "Considers upscale dining indulgent and unnecessary.",
        "Books cheap or moderate restaurants regardless of budget.",
    ],
    ("Restaurant", "rating", ">=", "Q2"): [
        "Doesn't weight restaurant ratings; books regardless of reviews.",
        "Comfortable at modestly-rated eateries below the median.",
        "Skips the review-filter step on dining.",
        "Eats at low-rated restaurants without concern.",
        "Doesn't insist on median-or-better ratings on meals.",
    ],
    ("Restaurant", "rating", ">=", "Q3"): [
        "Doesn't chase top-rated restaurants; mid-range is fine.",
        "Avoids fine-dining acclaim; prefers casual, ordinary spots.",
        "Skips the top-of-rating-curve step on meals.",
        "Considers top-tier review scores overkill on dining.",
        "Comfortable in moderately-rated restaurants.",
    ],
    ("Restaurant", "rating", "<=", "Q3"): [
        "Insists on top-rated restaurants; won't settle for moderate reviews.",
        "Books only the highest-rated dining; mid-tier is a non-starter.",
        "Demands review-vetted restaurants at the very top of the curve.",
        "Won't eat anywhere below the acclaimed tier.",
        "Picks meals exclusively from the most-reviewed restaurants.",
    ],

    ("Attraction", "rating", ">=", "Q2"): [
        "Doesn't weight attraction ratings; visits regardless of reviews.",
        "Comfortable at modestly-rated venues below the median.",
        "Skips the review-filter step when planning sightseeing.",
        "Visits low-rated attractions without concern.",
        "Doesn't insist on median-or-better ratings on sights.",
    ],
    ("Attraction", "rating", ">=", "Q3"): [
        "Doesn't chase top-rated attractions; the modest ones are fine.",
        "Avoids the crowds at acclaimed venues; prefers quieter sights.",
        "Skips the top-of-rating-curve step on sightseeing.",
        "Considers top-tier review scores overkill on attractions.",
        "Comfortable visiting moderately-rated landmarks.",
    ],

    ("Transportation", "cost", "<=", "Q1"): [
        "Avoids the cheapest transit; values comfort while traveling.",
        "Won't settle for budget airlines or bus routes.",
        "Prefers paying more for faster, more comfortable transport.",
        "Believes transit is worth investing in.",
        "Books mid- or upper-tier transportation even on tight budgets.",
    ],
    ("Transportation", "cost", ">=", "Q3"): [
        "Never splurges on transit; finds premium classes wasteful.",
        "Avoids first-class travel; sticks to economy.",
        "Won't pay top-bracket prices for transportation.",
        "Considers premium transit indulgent for short trips.",
        "Books cheap or standard transportation regardless of budget.",
    ],
}


# --------------------------------------------------------------------------- #
# Preferred Destinations  (anchored on destination archetype class)
# --------------------------------------------------------------------------- #

PREFERRED_DESTINATION_PHRASING = {
    "Beach Resorts": [
        "Drawn to coastal getaways with sand, sun, and ocean access.",
        "Loves beach resorts, oceanfront views, and seaside relaxation.",
        "Prefers beach destinations for warm-weather travel.",
        "Drawn to swimming, snorkeling, and beach lounging.",
        "Values salt-air mornings and oceanfront sunsets.",
        "Loves walking barefoot on warm sand and beach reading.",
        "Prefers beach resort itineraries with poolside lounging.",
        "Drawn to coastal villages and seaside boardwalks.",
        "Loves the soundtrack of waves and the warmth of sun.",
        "Prefers tropical or temperate beach destinations.",
    ],
    "Coastal Areas": [
        "Drawn to working harbors, fishing villages, and coastal towns.",
        "Loves seaside towns with maritime culture and history.",
        "Prefers walkable coastal destinations with character.",
        "Drawn to lighthouse tours and shoreline drives.",
        "Values salt-air mornings and coastal hiking trails.",
        "Loves cliff-top views and tidal pool exploration.",
        "Prefers coastal towns with authentic local character.",
        "Drawn to historic seaports and fishing communities.",
        "Loves coastal walks, shore birds, and lighthouse visits.",
        "Prefers Atlantic, Pacific, or Gulf coastal destinations.",
    ],
    "Caribbean Islands": [
        "Drawn to tropical Caribbean destinations and island culture.",
        "Loves island time, Caribbean cuisine, and reggae rhythms.",
        "Prefers tropical island vacations with palm-tree beaches.",
        "Drawn to Caribbean island hopping and resort culture.",
        "Values warm winters spent on tropical islands.",
        "Loves snorkeling reefs and Caribbean watersports.",
        "Prefers all-inclusive Caribbean resorts for full immersion.",
        "Drawn to Caribbean local flavors — jerk chicken, fresh fish.",
        "Loves the relaxed pace of Caribbean island time.",
        "Prefers tropical island settings for unwinding vacations.",
    ],
    "National Parks": [
        "Drawn to national parks for hiking and wilderness immersion.",
        "Loves the grandeur of America's national park system.",
        "Prefers nature-focused itineraries with park visits.",
        "Drawn to canyon hiking, mountain trails, and waterfall photography.",
        "Values nights under dark skies in national parks.",
        "Loves wildlife observation in park ecosystems.",
        "Prefers multi-park road-trip itineraries.",
        "Drawn to ranger talks and visitor-center exhibits.",
        "Loves the variety of US national park landscapes.",
        "Prefers wilderness lodges and rustic park accommodations.",
    ],
    "Major Cities": [
        "Drawn to major metropolitan destinations for urban energy.",
        "Loves big-city culture — museums, food scenes, nightlife.",
        "Prefers urban itineraries with neighborhoods to explore.",
        "Drawn to skyline views, public transit, and street life.",
        "Values the variety and density of major cities.",
        "Loves big-city walking tours and museum-hopping.",
        "Prefers urban destinations with strong food and arts scenes.",
        "Drawn to global metropolises for cultural immersion.",
        "Loves city neighborhoods and their distinct characters.",
        "Prefers itineraries centered on major-city exploration.",
    ],
    "Historic Cities": [
        "Drawn to historic cities with preserved architecture.",
        "Loves walking centuries-old streets and historic districts.",
        "Prefers cities with deep historical layers.",
        "Drawn to colonial-era towns and preservation districts.",
        "Values the storytelling of historic neighborhoods.",
        "Loves cathedral towns and medieval city centers.",
        "Prefers itineraries focused on heritage and history.",
        "Drawn to cobblestone streets and historic markers.",
        "Loves docent-led tours of historic city centers.",
        "Prefers cities where history is woven into daily life.",
    ],
    "Family-friendly Resorts": [
        "Drawn to family-friendly resorts with kid amenities.",
        "Loves multi-generational resort vacations.",
        "Prefers all-inclusive family destinations with kids' programs.",
        "Drawn to resorts with pools, water parks, and activities.",
        "Values family-oriented dining and entertainment options.",
        "Loves resort destinations that include child care.",
        "Prefers resort itineraries with built-in family amenities.",
        "Drawn to themed resorts that appeal across generations.",
        "Loves the convenience of all-inclusive family stays.",
        "Prefers family resorts with kid clubs and teen programs.",
    ],
    "Quiet Countryside": [
        "Drawn to quiet countryside destinations for relaxation.",
        "Loves rural retreats and farmhouse stays.",
        "Prefers slower-paced itineraries in pastoral settings.",
        "Drawn to countryside drives and small-town discoveries.",
        "Values rural quiet and farm-fresh meals.",
        "Loves country roads and rolling-hills landscapes.",
        "Prefers off-the-beaten-path countryside destinations.",
        "Drawn to vineyards, orchards, and country-fair traditions.",
        "Loves the peace of small villages and rural towns.",
        "Prefers countryside getaways away from urban density.",
    ],
    "Design Capitals": [
        "Drawn to cities at the forefront of design, architecture, and visual culture.",
        "Loves contemporary design districts, galleries, and architecture trails.",
        "Appreciates cities known for design schools, ateliers, and creative scenes.",
        "Picks destinations where design is part of the city's identity.",
        "Drawn to fashion houses, design studios, and architectural landmarks.",
        "Loves the energy of design-focused cities with strong cultural output.",
        "Appreciates the intersection of art, fashion, and architecture.",
        "Picks destinations known for design events and creative communities.",
        "Drawn to cities with major design and architecture museums.",
        "Loves cities where design shapes everyday spaces.",
    ],
    "Exotic Islands": [
        "Drawn to far-off island destinations with unique culture and landscapes.",
        "Loves the appeal of tropical islands and remote archipelagos.",
        "Picks distinctive island destinations for memorable getaways.",
        "Drawn to volcanic islands, lagoons, and isolated coves.",
        "Loves the sense of escape that exotic island travel brings.",
        "Appreciates the cultural distinctiveness of island destinations.",
        "Picks islands for their snorkeling, diving, or beach culture.",
        "Drawn to island destinations with strong cultural traditions.",
        "Loves the slow pace and natural beauty of island getaways.",
        "Picks islands as a once-in-a-while travel highlight.",
    ],
    "Historical Sites": [
        "Drawn to itineraries built around major historical landmarks.",
        "Loves visiting battlefields, monuments, and heritage sites.",
        "Picks destinations with rich, well-preserved historical layers.",
        "Drawn to colonial-era sites, ancient ruins, and heritage trails.",
        "Loves history-museum walks and guided tours of significant sites.",
        "Picks destinations for their UNESCO and national heritage markers.",
        "Drawn to towns with documented historical roles.",
        "Loves the depth that historical-site visits add to a trip.",
        "Picks itineraries that anchor each day around a historical landmark.",
        "Drawn to commemorative sites and history-preservation projects.",
    ],
    "Luxury Resorts": [
        "Drawn to high-end resort getaways with full-service amenities.",
        "Loves boutique luxury properties and signature service.",
        "Picks destinations with premier resort options.",
        "Drawn to spa-anchored, all-inclusive luxury stays.",
        "Loves the polish of five-star resort experiences.",
        "Picks resort destinations for the quality of the stay itself.",
        "Drawn to private-island and beachfront luxury properties.",
        "Loves luxury-resort culinary, spa, and pool experiences.",
        "Picks resort destinations where the property is the destination.",
        "Drawn to flagship luxury resort brands and signature hospitality.",
    ],
    "Music Festivals": [
        "Drawn to destinations known for music festivals and live shows.",
        "Loves traveling to catch specific concerts or festival lineups.",
        "Picks cities for their music venues and seasonal events.",
        "Drawn to jazz clubs, indie venues, and festival main stages.",
        "Loves the energy of festival-going crowds and outdoor stages.",
        "Picks travel dates to align with music events.",
        "Drawn to cities with strong residency or touring-act calendars.",
        "Loves the cultural mix of music-anchored travel.",
        "Picks destinations known for distinct music scenes (jazz, country, electronic).",
        "Drawn to multi-day festival passes and music-focused itineraries.",
    ],
    "Remote Locations": [
        "Drawn to off-the-grid destinations away from tourist density.",
        "Loves remote cabin retreats, fishing lodges, and back-country travel.",
        "Picks destinations where seclusion is part of the appeal.",
        "Drawn to wilderness camps and rural inns.",
        "Loves the stillness of low-traffic, remote-area travel.",
        "Picks remote destinations to disconnect from city pace.",
        "Drawn to small-town America and lightly-populated regions.",
        "Loves long drives between remote stops.",
        "Picks destinations where solitude is built into the experience.",
        "Drawn to places that take effort to reach.",
    ],
    "Tech Conferences": [
        "Drawn to cities hosting major tech conferences and meetups.",
        "Loves travel that coincides with industry events.",
        "Picks destinations to align with conference calendars.",
        "Drawn to tech-hub cities for product launches and demos.",
        "Loves the side-events and after-parties of major conferences.",
        "Picks tech-conference travel as both work and exploration.",
        "Drawn to cities with strong startup and venture ecosystems.",
        "Loves combining conference attendance with city tourism.",
        "Picks travel timing around developer conferences and expos.",
        "Drawn to cities where the tech industry shapes the calendar.",
    ],
    "Business Hubs": [
        "Drawn to major financial and corporate cities.",
        "Loves the energy of business-district skyscrapers and boardroom lunches.",
        "Picks cities known as commercial and financial centers.",
        "Drawn to global business hubs and trading-floor cities.",
        "Loves the polish of corporate-cluster downtowns.",
        "Picks destinations with active business-travel ecosystems.",
        "Drawn to financial-district hotels and convention-center proximity.",
        "Loves cities where commerce and culture intersect.",
        "Picks travel that mixes business meetings with city exploration.",
        "Drawn to cities ranked among global business centers.",
    ],
}


PREFERRED_DESTINATION_PHRASING_INVERSIONS = {
    "Beach Resorts": [
        "Avoids beach destinations; finds them too crowded or one-note.",
        "Doesn't enjoy sun, sand, and beach culture.",
        "Prefers cultural travel to beach resort itineraries.",
        "Finds beach resorts repetitive and surface-level.",
        "Avoids tropical beach destinations.",
    ],
    "Coastal Areas": [
        "Avoids coastal destinations; prefers inland travel.",
        "Doesn't enjoy seaside towns or maritime culture.",
        "Prefers landlocked destinations over coastal areas.",
        "Finds coastal travel monotonous.",
        "Avoids harbor towns and fishing villages.",
    ],
    "Caribbean Islands": [
        "Avoids Caribbean destinations; prefers mainland travel.",
        "Doesn't enjoy island culture or tropical heat.",
        "Prefers temperate-climate destinations over Caribbean islands.",
        "Finds Caribbean resorts touristy and over-marketed.",
        "Avoids tropical islands as travel choices.",
    ],
    "National Parks": [
        "Avoids national parks; prefers urban travel.",
        "Doesn't enjoy wilderness immersion.",
        "Prefers cultural destinations to nature parks.",
        "Finds park visits tedious and physically demanding.",
        "Avoids outdoor-focused itineraries.",
    ],
    "Major Cities": [
        "Avoids major cities; prefers quieter destinations.",
        "Doesn't enjoy urban density and metropolitan energy.",
        "Prefers small towns to big cities.",
        "Finds major cities overstimulating.",
        "Avoids metropolitan travel.",
    ],
    "Historic Cities": [
        "Avoids historic cities; prefers contemporary destinations.",
        "Doesn't enjoy heritage-focused travel.",
        "Prefers modern cities to historic districts.",
        "Finds historic-city tourism over-curated.",
        "Avoids preservation-focused itineraries.",
    ],
    "Family-friendly Resorts": [
        "Avoids family resorts; prefers adult travel.",
        "Doesn't enjoy resort culture or kid amenities.",
        "Prefers boutique stays to family-oriented resorts.",
        "Finds family resorts crowded and chaotic.",
        "Avoids multi-generational resort destinations.",
    ],
    "Quiet Countryside": [
        "Avoids quiet countryside; prefers urban energy.",
        "Doesn't enjoy rural retreats; needs more stimulation.",
        "Prefers cities to countryside destinations.",
        "Finds rural itineraries dull and slow-paced.",
        "Avoids countryside travel.",
    ],
    "Design Capitals": [
        "Finds design-focused cities pretentious or inaccessible.",
        "Avoids design-district itineraries; prefers more organic city exploration.",
        "Doesn't enjoy design-museum-heavy trips.",
        "Sensitive to design-snobby atmospheres; prefers grounded destinations.",
        "Considers design tourism niche; not a draw.",
    ],
    "Exotic Islands": [
        "Finds island destinations too remote or limited in activity.",
        "Avoids overwater-bungalow itineraries; prefers continental travel.",
        "Doesn't enjoy island travel logistics — getting to and around them.",
        "Prefers mainland destinations with broader cultural depth.",
        "Considers island trips too repetitive or single-themed.",
    ],
    "Historical Sites": [
        "Finds historical-site touring repetitive or text-heavy.",
        "Avoids itineraries dominated by museums and monuments.",
        "Doesn't enjoy guided history tours; prefers free exploration.",
        "Prefers contemporary culture over historical preservation.",
        "Considers heritage sites less engaging than living-city experiences.",
    ],
    "Luxury Resorts": [
        "Finds luxury resorts isolated from the destination's character.",
        "Avoids all-inclusive luxury formats; prefers local exploration.",
        "Doesn't enjoy resort culture; prefers urban or rural authenticity.",
        "Considers luxury-resort spending wasteful for a short trip.",
        "Prefers character-rich smaller stays over signature properties.",
    ],
    "Music Festivals": [
        "Finds festival crowds overwhelming and exhausting.",
        "Avoids destinations where music festivals dominate the calendar.",
        "Doesn't enjoy concert-anchored travel; prefers quieter itineraries.",
        "Sensitive to festival logistics and crowd density.",
        "Considers music-festival travel niche; not a draw.",
    ],
    "Remote Locations": [
        "Finds remote-area travel logistically difficult or isolating.",
        "Avoids destinations far from major airports.",
        "Doesn't enjoy remote-area accommodations; prefers urban stays.",
        "Prefers destinations with active scenes and services nearby.",
        "Considers remote travel too quiet or too sparsely-amenitied.",
    ],
    "Tech Conferences": [
        "Finds tech-conference travel exhausting; prefers personal trips.",
        "Avoids tech-anchored destinations during conference weeks.",
        "Doesn't enjoy conference-density city visits.",
        "Prefers travel disconnected from professional events.",
        "Considers conference-driven travel a chore, not a draw.",
    ],
    "Business Hubs": [
        "Finds business-cluster cities sterile and impersonal.",
        "Avoids destinations dominated by corporate culture.",
        "Doesn't enjoy business-traveler urban environments.",
        "Prefers cities with stronger residential or cultural identity.",
        "Considers business-hub travel uninteresting outside work.",
    ],
}
