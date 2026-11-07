"""Additional facial presets and an animatable skin-attached cheek blush."""
import math

BLUSH_CONTROL = 'Ellen_CheekBlush'
BLUSH_ATTRIBUTE = 'Ellen_CheekBlushMask'
DISGUST_CONTROL = 'Ellen_DisgustFrown'
GRIMACE_CONTROL = 'Ellen_DisgustGrimace'
REFERENCE_EYES = 'Ellen_DisgustReferenceEyes'
REFERENCE_BROWS = 'Ellen_DisgustReferenceBrows'
REFERENCE_MOUTH = 'Ellen_DisgustReferenceMouth'
REFERENCE_CONTROLS = (REFERENCE_EYES, REFERENCE_BROWS, REFERENCE_MOUTH)
PAIR_CONTROLS = ('Ellen_DisgustSmallOpen', 'Ellen_DisgustCurvedTeeth')
SIDE_GLANCE_CONTROL = 'Ellen_ReferenceSideGlance'
SOFT_SMILE_CONTROL = 'Ellen_ReferenceSoftSmile'
BLANK_BROWS_CONTROL = 'Ellen_ReferenceBlankBrows'
BELLE_REFERENCE_PRESET = 'belle_weary'
BELLE_WEARY_BROWS = 'Belle_WearyReferenceBrows'
SELECTED_REFERENCE_PRESETS = ('relaxed_listen', 'side_glance', 'playful_wink', 'blank_stare')
TEETH_PRESETS = frozenset(('laugh', 'disgust', 'disgust_curved'))
EXTRA_PRESETS = {
    'happy_closed': ('弯眼甜笑', {'笑い': 1.0, 'にこり': .78, '上': .28, '口角上げ': .68,
                               '口横狭め': .08}),
    'disgust': ('嫌弃 / 小开口', {PAIR_CONTROLS[0]: 1.0, 'Ellen_ShowTeeth': 1.0,
                                 'Ellen_ForeheadShadow': .60}),
    'shy': ('脸红害羞', {'じと目': .2, 'まばたき': .12, '困る': .48, 'にこり': .2,
                       '上': .18, '口角上げ': .2, '口横狭め': .2, BLUSH_CONTROL: .85}),
    'blush_smile': ('脸红闭眼笑', {'笑い': 1.0, 'にこり': .65, '上': .23,
                                  '口角上げ': .42, BLUSH_CONTROL: .75}),
    'embarrassed': ('害羞 / 窘迫', {'じと目': .42, '困る': .95, '上': .3, '下': .12,
                                    'ウィンク２右': .20, '口角下げ': .3, '口横狭め': .15, 'あ': .08, BLUSH_CONTROL: 1.0}),
    'smug': ('得意 / 坏笑', {'じと目': .5, 'にこり': .4, '上左': .70,
                              '上右': .1, '口角上げ': .78, '口横狭め': .08}),
    'surprised': ('惊讶 / 愣住', {'びっくり左': .7, 'びっくり右': .7, '上': .8,
                                  '困る': .12, 'お': .3}),
    'pout': ('委屈 / 小生气', {'じと目': .4, '怒り': .68, '困る': .25, '下': .42,
                                'う': .3, '口角下げ': .48, '口横狭め': .15}),
    'disgust_curved': ('嫌弃 / 弧形露牙', {PAIR_CONTROLS[1]: 1.0, 'Ellen_ShowTeeth': 1.0,
                                       'Ellen_ForeheadShadow': .60}),
    'relaxed_listen': ('参考01 / 慵懒听你说', {'じと目': .45, 'まばたき': .025,
                         'にこり': .45, '上': .12, SOFT_SMILE_CONTROL: .38, '口横狭め': .06}),
    'side_glance': ('参考02 / 半眯眼侧视', {'じと目': 1.0, 'まばたき': .48,
                       '真面目': .18, '上左': .20, '上右': .04,
                       '口角上げ': .10, SIDE_GLANCE_CONTROL: 1.0}),
    'playful_wink': ('参考09 / 调皮眨眼', {'ウィンク２右': 1.0, 'にこり': .60,
                         '上': .13, '口角上げ': .28, 'ワ': .52}),
    'blank_stare': ('新参考 / 呆萌小开口', {'じと目': .15, BLANK_BROWS_CONTROL: 1.0,
                        'お': .085, '口横狭め': .12}),
    BELLE_REFERENCE_PRESET: ('铃 / 半眯眼坏笑', {'じと目': .85, 'まばたき': .18,
                        'にこり': .20, '上': .06, '口角下げ': .16, '口角上げ': .14,
                        '口横狭め': .40, BLUSH_CONTROL: .30}),
}

EXTRA_PRESETS['blank_stare_shy'] = ('呆萌 / 微红害羞', {
    **EXTRA_PRESETS['blank_stare'][1], BLUSH_CONTROL: .65})

MOUTH_PROFILES = {
    'happy_closed': (.9, .25, .85), 'disgust': (.82, 0, .8),
    'shy': (.85, .1, .82), 'blush_smile': (.9, .25, .85),
    'embarrassed': (.85, 0, .82), 'smug': (.88, .16, .85),
    'surprised': (1.0, 0, .95), 'pout': (.85, 0, .82),
    'disgust_curved': (.82, 0, .8),
    'relaxed_listen': (.90, .08, .85), 'side_glance': (.88, 0, .85),
    'playful_wink': (.90, .20, .85), 'blank_stare': (.95, 0, .90),
    BELLE_REFERENCE_PRESET: (.90, .12, .85),
}
MOUTH_PROFILES['blank_stare_shy'] = MOUTH_PROFILES['blank_stare']


def ensure_belle_weary_brows(mesh):
    """Flatten Belle's brow arch and lift the inner ends for the weary reference."""
    keys = mesh.data.shape_keys
    if BELLE_WEARY_BROWS in keys.key_blocks:
        return
    if 'Belle_MouthWidth_Adjustment' not in keys.key_blocks:
        raise ValueError('此参考眉形仅为铃模型配置')
    labels = [m.get('original_material', m.name.rsplit('|', 1)[-1].strip().split('.')[0])
              if m else '' for m in mesh.data.materials]
    ids = {v for p in mesh.data.polygons if labels[p.material_index] == '眉' for v in p.vertices}
    if not ids or '困る' not in keys.key_blocks:
        raise ValueError('缺少铃的眉毛网格或原始眉形')
    rest = keys.key_blocks[0].data
    inner = min(abs(rest[i].co.x) for i in ids)
    outer = max(abs(rest[i].co.x) for i in ids)
    key = mesh.shape_key_add(name=BELLE_WEARY_BROWS, from_mix=False)
    for i in ids:
        t = (abs(rest[i].co.x) - inner) / (outer - inner)
        key.data[i].co = rest[i].co + (keys.key_blocks['困る'].data[i].co - rest[i].co) * .55
        key.data[i].co.z += .003 * (1 - t) - .001 * t
        # Keep the moved brow surface in front of the forehead skin.
        key.data[i].co.y -= .0015
    key.value = 0


def ensure_reference_side_glance(mesh):
    """A small iris-only offset, independent of captured eyelids and eye bones."""
    keys = mesh.data.shape_keys
    if SIDE_GLANCE_CONTROL in keys.key_blocks:
        return
    slots = {i for i, material in enumerate(mesh.data.materials)
             if material and material.name.rsplit('|', 1)[-1].strip() == '目'}
    iris_ids = {v for polygon in mesh.data.polygons if polygon.material_index in slots
                for v in polygon.vertices}
    if not iris_ids:
        raise ValueError('未找到艾莲独立的虹膜材质，无法创建侧视表情')
    rest = keys.key_blocks[0].data
    key = mesh.shape_key_add(name=SIDE_GLANCE_CONTROL, from_mix=False)
    for index in iris_ids:
        key.data[index].co = rest[index].co
        key.data[index].co.x += .003
        key.data[index].co.y -= .00035
    key.value = 0


def ensure_reference_blank_brows(mesh):
    """Raise the inner brow ends and lower the outer ends on both sides."""
    keys = mesh.data.shape_keys
    if BLANK_BROWS_CONTROL in keys.key_blocks:
        return
    slots = {i for i, material in enumerate(mesh.data.materials)
             if material and material.name.rsplit('|', 1)[-1].strip() == '眉'}
    brow_ids = {v for polygon in mesh.data.polygons if polygon.material_index in slots
                for v in polygon.vertices}
    if not brow_ids:
        raise ValueError('未找到艾莲独立的眉毛材质，无法创建参考眉形')
    rest = keys.key_blocks[0].data
    key = mesh.shape_key_add(name=BLANK_BROWS_CONTROL, from_mix=False)
    for index in brow_ids:
        t = max(0, min(1, (abs(rest[index].co.x) - .015) / .045))
        key.data[index].co.z += .010 * (1 - t) - .005 * t
    key.value = 0


def ensure_reference_soft_smile(mesh):
    """Make the subtle authored mouth-corner lift readable in a closeup."""
    keys = mesh.data.shape_keys
    if SOFT_SMILE_CONTROL in keys.key_blocks:
        return
    rest = keys.key_blocks[0].data
    up = keys.key_blocks.get('口角上げ')
    if up is None:
        raise ValueError('缺少原始嘴角上扬形态，无法创建轻笑表情')
    key = mesh.shape_key_add(name=SOFT_SMILE_CONTROL, from_mix=False)
    for i, point in enumerate(key.data):
        basis = rest[i].co
        point.co = basis + (up.data[i].co - basis) * 3.0
    key.value = 0


def ensure_disgust_pair(mesh):
    """Separate the two references instead of blending incompatible mouth shapes."""
    keys = mesh.data.shape_keys
    if all(name in keys.key_blocks for name in PAIR_CONTROLS) and mesh.get('ellen_disgust_pair_revision', 0) >= 2:
        return
    rest = keys.key_blocks[0].data
    required = ('じと目', 'まばたき', '□', 'え３', 'い２')
    if any(name not in keys.key_blocks for name in required):
        raise ValueError('当前角色缺少艾莲原始嘴形和眼形')
    ids = {}
    for label in ('眉', '齿'):
        slots = {i for i, m in enumerate(mesh.data.materials)
                 if m and m.name.rsplit('|', 1)[-1].strip() == label}
        ids[label] = {v for p in mesh.data.polygons if p.material_index in slots for v in p.vertices}
    for index, name in enumerate(PAIR_CONTROLS):
        key = keys.key_blocks.get(name)
        if key is None:
            key = mesh.shape_key_add(name=name, from_mix=False)
        old_value = key.value
        blink, square, e3, bend = (.28, .22, .10, .001) if index == 0 else (.32, .035, 0, .008)
        for i, point in enumerate(key.data):
            co = rest[i].co.copy()
            x, y, z = co
            if z > 1.325 and i not in ids['眉']:
                for source, weight in (('じと目', .9), ('まばたき', blink)):
                    co += (keys.key_blocks[source].data[i].co-rest[i].co)*weight
            if i in ids['眉']:
                t = max(0, min(1, (abs(x)-.015)/.045))
                co.z += .008*(1-t)**1.8-.002*t+.0008*math.sin(math.pi*t)
                co.x -= math.copysign(.0006*(1-t), x)
            if z < 1.332:
                fade = math.exp(-(x/.029)**4-((z-1.301)/.018)**2)
                fade *= max(0, min(1, (-.01-y)/.02))
                co += ((keys.key_blocks['□'].data[i].co-rest[i].co)*square
                       +(keys.key_blocks['え３'].data[i].co-rest[i].co)*e3
                       +(keys.key_blocks['い２'].data[i].co-rest[i].co)*(.5 if index else 0))*fade
                co.z += (bend*.3-bend*(x/.018)**2+.0015*x/.02)*fade
                if index == 0:
                    co.x -= co.x*.12*fade
                elif i in ids['齿']:
                    co.z = 1.301+(co.z-1.301)*.8
                if index == 1:
                    co.x += co.x*.08*fade
            point.co = co
        key.value = old_value
    mesh['ellen_disgust_pair_revision'] = 2


def ensure_reference_disgust(mesh):
    """Three localized additive sculpts; authored keys and Basis stay unchanged."""
    keys = mesh.data.shape_keys
    if all(name in keys.key_blocks for name in REFERENCE_CONTROLS) and mesh.get('ellen_disgust_mouth_revision', 0) >= 2:
        return
    required = ('い２', 'え３', 'じと目', '怒り目２', '眼角下', 'まばたき')
    if any(name not in keys.key_blocks for name in required):
        raise ValueError('当前角色缺少艾莲面部形态，无法创建参考嫌弃脸')
    slots = {label: {i for i, m in enumerate(mesh.data.materials)
                    if m and m.name.rsplit('|', 1)[-1].strip() == label}
             for label in ('眉', '齿')}
    ids = {label: {v for p in mesh.data.polygons if p.material_index in indices for v in p.vertices}
           for label, indices in slots.items()}
    if not ids['眉'] or not ids['齿']:
        raise ValueError('未找到独立眉毛或牙齿材质')
    rest = keys.key_blocks[0].data
    for name in REFERENCE_CONTROLS:
        if name in keys.key_blocks and name != REFERENCE_MOUTH:
            continue
        key = keys.key_blocks.get(name)
        if key is None:
            key = mesh.shape_key_add(name=name, from_mix=False)
        for i, point in enumerate(key.data):
            co = rest[i].co.copy()
            x, y, z = co
            if name == REFERENCE_EYES and z > 1.325 and i not in ids['眉']:
                side_weight = .55 if x < 0 else 1.0
                for source, weight in (('じと目', .7), ('怒り目２', .1),
                                       ('眼角下', .2), ('まばたき', .18)):
                    co += (keys.key_blocks[source].data[i].co-rest[i].co)*weight*side_weight
            elif name == REFERENCE_BROWS and i in ids['眉']:
                t = max(0, min(1, (abs(x)-.015)/.045))
                co.z += .014*(1-t)**2.2-.003*t-.002*math.sin(math.pi*t)
                co.x -= math.copysign(.0015*(1-t), x)
            elif name == REFERENCE_MOUTH and z < 1.332:
                fade = math.exp(-(x/.029)**4-((z-1.301)/.018)**2)
                fade *= max(0, min(1, (-.01-y)/.02))
                co += ((keys.key_blocks['い２'].data[i].co-rest[i].co)*.6
                       +(keys.key_blocks['え３'].data[i].co-rest[i].co)*.10)*fade
                co.x += .0015*fade
                co.z += (1.301-co.z)*.25*fade
                co.z += .003*(x/.022)*fade
            point.co = co
        key.value = 0
    mesh['ellen_disgust_mouth_revision'] = 2


def ensure_disgust_grimace(mesh):
    """Narrow, downturned tooth-bearing mouth, separate from captured vowels."""
    keys = mesh.data.shape_keys
    if GRIMACE_CONTROL in keys.key_blocks:
        return
    rest = keys.key_blocks[0]
    down = keys.key_blocks.get('口角下げ')
    clenched = keys.key_blocks.get('い２')
    if down is None or clenched is None:
        raise ValueError('当前角色缺少咧嘴形态，无法创建露牙嫌弃嘴形')
    key = mesh.shape_key_add(name=GRIMACE_CONTROL, from_mix=False)
    for index, point in enumerate(key.data):
        basis = rest.data[index].co
        point.co = basis + (down.data[index].co-basis)*2.8 + (clenched.data[index].co-basis)*.7
    key.value = 0


def ensure_disgust_control(mesh):
    """An additive, closed-lip frown; keep all authored facial keys untouched."""
    keys = mesh.data.shape_keys
    if DISGUST_CONTROL in keys.key_blocks:
        return
    rest = keys.key_blocks[0]
    down = keys.key_blocks.get('口角下げ')
    wide = keys.key_blocks.get('口横広げ')
    if down is None or wide is None:
        raise ValueError('当前角色缺少嘴角形态，无法创建嫌弃嘴形')
    key = mesh.shape_key_add(name=DISGUST_CONTROL, from_mix=False)
    for index, point in enumerate(key.data):
        basis = rest.data[index].co
        point.co = basis + (down.data[index].co-basis)*2.5 + (wide.data[index].co-basis)*.2
    key.value = 0


def ensure_blush_control(mesh):
    """Only tint facial skin; a zero-value control leaves original shading intact."""
    keys = mesh.data.shape_keys
    if BLUSH_CONTROL not in keys.key_blocks:
        mesh.shape_key_add(name=BLUSH_CONTROL, from_mix=False).value = 0
    labels = [m.get('original_material', m.name.rsplit('|', 1)[-1].strip().split('.')[0])
              if m else '' for m in mesh.data.materials]
    skin = {i for i, label in enumerate(labels) if label in {'颜', '颜2'}}
    brows = {i for i, label in enumerate(labels) if label == '眉'}
    brow_ids = {v for p in mesh.data.polygons if p.material_index in brows for v in p.vertices}
    if not skin or not brow_ids:
        raise ValueError('当前角色没有可定位的脸部皮肤或眉毛材质，无法添加脸红')
    rest = keys.key_blocks[0].data
    brow_height = sum(rest[v].co.z for v in brow_ids)/len(brow_ids)
    brow_width = max(abs(rest[v].co.x) for v in brow_ids)
    scale = brow_width/.0596
    center_z = brow_height-.046*scale
    front = min(rest[v].co.y for p in mesh.data.polygons if p.material_index in skin for v in p.vertices)
    skin_ids = {v for p in mesh.data.polygons if p.material_index in skin for v in p.vertices}
    attribute = mesh.data.attributes.get(BLUSH_ATTRIBUTE)
    if attribute is None:
        attribute = mesh.data.attributes.new(BLUSH_ATTRIBUTE, 'FLOAT', 'POINT')
    values = []
    for i, point in enumerate(rest):
        x, y, z = point.co
        distance = ((abs(x)-.042*scale)/(.025*scale))**2+((z-center_z)/(.014*scale))**2
        front_fade = max(0, min(1, (front+.045*scale-y)/(.02*scale)))
        values.append(math.exp(-distance*1.5)*front_fade if i in skin_ids else 0)
    attribute.data.foreach_set('value', values)
    for index in skin:
        material = mesh.data.materials[index]
        if material.get('ellen_blush_key') == keys.name:
            continue
        if material.node_tree.nodes.get('Ellen Cheek Blush Tint'):
            raise ValueError('脸红材质属于另一个面部控制，请先建立独立材质副本')
        material = material.copy()
        material.name = 'Ellen Blush | '+mesh.data.materials[index].name.rsplit('|', 1)[-1].strip()
        target = next((n for n in material.node_tree.nodes if n.type == 'GROUP'
                       and n.inputs.get('DData') and n.inputs['DData'].is_linked), None)
        if not target:
            raise ValueError('脸部材质缺少 DData 输入，未改动材质')
        material['ellen_blush_key'] = keys.name
        mesh.data.materials[index] = material
        nodes, links = material.node_tree.nodes, material.node_tree.links
        source = target.inputs['DData'].links[0].from_socket
        mask = nodes.new('ShaderNodeAttribute')
        mask.name, mask.attribute_name = 'Ellen Cheek Blush Mask', BLUSH_ATTRIBUTE
        value = nodes.new('ShaderNodeValue')
        value.name, value.label = 'Ellen Cheek Blush Strength', '脸红强度'
        driver = value.outputs[0].driver_add('default_value').driver
        variable = driver.variables.new()
        variable.name, variable.type = 'blush', 'SINGLE_PROP'
        variable.targets[0].id_type, variable.targets[0].id = 'KEY', keys
        variable.targets[0].data_path = keys.key_blocks[BLUSH_CONTROL].path_from_id('value')
        driver.expression = 'blush * 0.64'
        amount = nodes.new('ShaderNodeMath')
        amount.operation, amount.name = 'MULTIPLY', 'Ellen Cheek Blush Amount'
        tint = nodes.new('ShaderNodeMixRGB')
        tint.name, tint.label = 'Ellen Cheek Blush Tint', '双颊红晕'
        tint.blend_type = 'MIX'
        tint.inputs[2].default_value = (.72, .105, .14, 1)
        links.new(mask.outputs['Fac'], amount.inputs[0])
        links.new(value.outputs[0], amount.inputs[1])
        links.new(amount.outputs[0], tint.inputs[0])
        links.new(source, tint.inputs[1])
        links.new(tint.outputs[0], target.inputs['DData'])
        mask.location, value.location = (-800, -700), (-800, -900)
        amount.location, tint.location = (-550, -700), (-200, -100)
    mesh['ellen_blush_revision'] = 1
